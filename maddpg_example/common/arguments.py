import argparse


def get_args(
        env_name="mpe",
        scenario_name="occupy",
        is_wind=False,
        num_agents=2,

        latent_dim=16,
        hidden_dim_act=32,
        hidden_dim_critic=64,
        hidden_dim_en=32,
        max_buffer_size=100000,
        actor_lr=1e-4,
        critic_lr=1e-3,
        encoder_lr=5e-4,
        kl_lambda=0.1,

        max_step=100,
        is_train=True,
        is_display=False,
        load_models_path=None,
        load_buffers_path=None,

        gamma=0.99,
        batch_size=512,
        num_episodes=60000,
        train_rate=2,
        print_rate=100,
        save_rate=5000,
        save_buffer_rate=10000,
        save_models_path=None,
        save_figures_path=None,
        save_buffers_path=None,

        device="cpu"
):
    parser = argparse.ArgumentParser("Task Unseen MARL")
    parser.add_argument("--env-name", type=str, default=env_name)
    parser.add_argument("--scenario-name", type=str, default=scenario_name)
    parser.add_argument("--is-wind", type=bool, default=is_wind)
    parser.add_argument("--num-agents", type=int, default=num_agents)

    parser.add_argument("--latent-dim", type=int, default=latent_dim)
    parser.add_argument("--hidden-dim-act", type=int, default=hidden_dim_act)
    parser.add_argument("--hidden-dim-critic", type=int, default=hidden_dim_critic)
    parser.add_argument("--hidden-dim-en", type=int, default=hidden_dim_en)
    parser.add_argument("--max-buffer-size", type=int, default=max_buffer_size)
    parser.add_argument("--actor-lr", type=float, default=actor_lr)
    parser.add_argument("--critic-lr", type=float, default=critic_lr)
    parser.add_argument("--encoder-lr", type=float, default=encoder_lr)
    parser.add_argument("--kl-lambda", type=float, default=kl_lambda)

    parser.add_argument("--max-step", type=int, default=max_step)
    parser.add_argument("--is-train", type=bool, default=is_train)
    parser.add_argument("--is-display", type=bool, default=is_display)

    parser.add_argument("--gamma", type=float, default=gamma)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--num-episodes", type=int, default=num_episodes)
    parser.add_argument("--train-rate", type=int, default=train_rate)
    parser.add_argument("--print-rate", type=int, default=print_rate)
    parser.add_argument("--save-rate", type=int, default=save_rate)
    parser.add_argument("--save-buffer-rate", type=int, default=save_buffer_rate)

    parser.add_argument("--load-models-path", type=str, default=load_models_path)
    parser.add_argument("--load-buffers-path", type=str, default=load_buffers_path)
    parser.add_argument("--save-models-path", type=str, default=save_models_path)
    parser.add_argument("--save-figures-path", type=str, default=save_figures_path)
    parser.add_argument("--save-buffers-path", type=str, default=save_buffers_path)

    parser.add_argument("--device", type=str, default=device)

    args = parser.parse_args()

    return args








































