00
JOBID=$(sbatch --parsable slurm/00_gpu_check.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/gpu_check_${JOBID}.out

01
JOBID=$(sbatch --parsable slurm/01_container_check.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/container_check_${JOBID}.out

02
JOBID=$(sbatch --parsable slurm/02_nccl_check.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/nccl_check_${JOBID}.out

03a
JOBID=$(sbatch --parsable slurm/03a_check_hf_packages.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/check_hf_packages_${JOBID}.out

03b
JOBID=$(sbatch --parsable slurm/03b_install_hf_packages.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/install_hf_packages_${JOBID}.out

03c
JOBID=$(sbatch --parsable slurm/03c_test_hf_model_and_data.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/test_hf_model_and_data_${JOBID}.out

04
JOBID=$(sbatch --parsable slurm/04_evaluate_base_model_mmlu.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/evaluate_base_model_mmlu_${JOBID}.out

05a
JOBID=$(sbatch --parsable slurm/05a_train_lora_multinode.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/train_lora_multinode_${JOBID}.out

05b
JOBID=$(sbatch --parsable slurm/05b_train_lora_single_gpu_tuning.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/train_lora_single_gpu_tuning_${JOBID}.out

06
JOBID=$(sbatch --parsable slurm/06_evaluate_finetuned_model_mmlu.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/evaluate_finetuned_model_mmlu_${JOBID}.out

07
JOBID=$(sbatch --parsable slurm/07_benchmark_inference_throughput.sbatch) || exit 1
echo "Submitted job $JOBID"
while squeue -j "$JOBID" | grep -q "$JOBID"; do
  sleep 10
done
echo "Job $JOBID finished"
printf '\a'
cat logs/benchmark_inference_throughput_${JOBID}.out
