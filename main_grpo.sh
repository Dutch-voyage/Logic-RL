set -x
MODEL_PATH=~/models/Qwen2.5-0.5B
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICE=1
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files=data/kk/instruct/3ppl/train.parquet \
    data.val_files=data/kk/instruct/3ppl/test.parquet \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=400 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.resume_mode=disable \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='Qwen-0.5B' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=../ckpts/Qwen2.5-0.5B \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@ 2>&1 | tee grpo.log

    