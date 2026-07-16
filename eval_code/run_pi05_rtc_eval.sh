#!/usr/bin/env bash
# π0.5 + RTC Evaluation Runner
# Quick script to run various RTC evaluation configurations

set -euo pipefail

# Default configuration
CONFIG_NAME="pi05_libero"
CHECKPOINT_DIR="${CHUNKFLOW_CHECKPOINT:-}"
TASK_SUITE="libero_object"
NUM_TRIALS=5
OUTPUT_BASE="outputs/libero/pi05_rtc_eval"
MODE="rtc"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint-dir|--task-suite|--num-trials|--output-dir|--mode)
            if [ "$#" -lt 2 ] || [[ "$2" == --* ]]; then
                echo "$1 requires a value" >&2
                exit 2
            fi
            ;;
    esac

    case $1 in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --task-suite)
            TASK_SUITE="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --checkpoint-dir PATH    Checkpoint directory (required; defaults to CHUNKFLOW_CHECKPOINT)"
            echo "  --task-suite SUITE       LIBERO task suite (default: libero_object)"
            echo "  --num-trials N           Trials per task (default: 5)"
            echo "  --output-dir PATH        Output directory (default: outputs/libero/pi05_rtc_eval)"
            echo "  --mode MODE              Evaluation mode: baseline|rtc|ablation|all (default: rtc)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Run standard RTC evaluation"
            echo "  CHUNKFLOW_CHECKPOINT=checkpoints/pi05_libero/run $0 --mode rtc"
            echo ""
            echo "  # Run full ablation study"
            echo "  CHUNKFLOW_CHECKPOINT=checkpoints/pi05_libero/run $0 --mode ablation --num-trials 10"
            echo ""
            echo "  # Run on custom checkpoint"
            echo "  $0 --checkpoint-dir checkpoints/pi05_libero/run --mode all"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ -z "$CHECKPOINT_DIR" ]; then
    echo "--checkpoint-dir is required" >&2
    exit 2
fi

echo "=========================================="
echo "π0.5 + RTC Evaluation"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Task suite: $TASK_SUITE"
echo "Trials per task: $NUM_TRIALS"
echo "Output: $OUTPUT_BASE"
echo "Mode: $MODE"
echo "=========================================="
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Function to run evaluation
run_eval() {
    local overlap=$1
    local blending=$2
    local output_suffix=$3

    echo ""
    echo "Running: overlap=$overlap, blending=$blending"
    echo "Output: $OUTPUT_BASE/$output_suffix"
    echo ""

    python eval_code/pi05_rtc_libero_eval.py \
        --config "$CONFIG_NAME" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --overlap-size "$overlap" \
        --blending-method "$blending" \
        --task-suite-name "$TASK_SUITE" \
        --num-trials-per-task "$NUM_TRIALS" \
        --output-dir "$OUTPUT_BASE/$output_suffix" \
        --save-videos \
        --save-plots

    echo "✓ Completed: $output_suffix"
}

# Run based on mode
case $MODE in
    baseline)
        echo "Running baseline (no RTC)..."
        run_eval 0 none "baseline"
        ;;

    rtc)
        echo "Running standard RTC (O=4, linear)..."
        run_eval 4 linear "rtc_standard"
        ;;

    ablation)
        echo "Running full ablation study..."
        python eval_code/pi05_rtc_libero_eval.py \
            --config "$CONFIG_NAME" \
            --checkpoint-dir "$CHECKPOINT_DIR" \
            --task-suite-name "$TASK_SUITE" \
            --num-trials-per-task "$NUM_TRIALS" \
            --output-dir "$OUTPUT_BASE/ablation" \
            --run-ablation \
            --ablation-overlaps 0 2 4 8 \
            --ablation-blendings none linear cosine \
            --save-videos \
            --save-plots
        echo "✓ Completed ablation study"
        ;;

    all)
        echo "Running comprehensive evaluation..."

        # Baseline
        run_eval 0 none "baseline"

        # RTC variations
        run_eval 2 linear "rtc_o2_linear"
        run_eval 4 linear "rtc_o4_linear"
        run_eval 4 cosine "rtc_o4_cosine"
        run_eval 8 linear "rtc_o8_linear"

        # Generate comparison report
        echo ""
        echo "Generating comparison report..."
        python -c "
import json
import glob
import os

print('\n' + '='*80)
print('Evaluation Results Summary')
print('='*80)
print('{:<25} {:<15} {:<12} {:<12}'.format('Configuration', 'Success Rate', 'Bjump', 'HF_ratio'))
print('-'*80)

for path in sorted(glob.glob('$OUTPUT_BASE/*/evaluation_report.json')):
    name = os.path.basename(os.path.dirname(path))
    try:
        with open(path) as f:
            data = json.load(f)
        sr = data['evaluation_summary']['overall_success_rate']
        bjump = data['temporal_metrics'].get('Bjump', {}).get('mean', 0)
        hf = data['temporal_metrics'].get('HF_ratio', {}).get('mean', 0)
        print('{:<25} {:<15.3f} {:<12.4f} {:<12.4f}'.format(name, sr, bjump, hf))
    except Exception as e:
        print('{:<25} Error: {}'.format(name, e))

print('='*80)
print(f'\nDetailed reports saved to: $OUTPUT_BASE/')
print('='*80)
"
        ;;

    *)
        echo "Error: Unknown mode: $MODE"
        echo "Valid modes: baseline, rtc, ablation, all"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_BASE"
echo ""
echo "View results:"
echo "  - JSON reports: $OUTPUT_BASE/*/evaluation_report.json"
echo "  - Videos: $OUTPUT_BASE/*/videos/"
echo "  - Plots: $OUTPUT_BASE/*/plots/"
echo ""
