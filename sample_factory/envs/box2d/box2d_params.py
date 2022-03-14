def box2d_override_defaults(env, parser):
    # Hyperparameters from https://arxiv.org/pdf/2111.02202.pdf
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_box2d',
        gamma=0.99,
        hidden_size=512,
        encoder_extra_fc_layers=0,
        exploration_loss_coeff=0.001,
        ppo_epochs=10,
        ppo_clip_value=0.2,
        env_frameskip=1,
        nonlinearity='tanh',
        # reward_scale=0.1,
    )


# noinspection PyUnusedLocal
def add_mujoco_env_args(env, parser):
    p = parser
