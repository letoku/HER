from utils.setup import her_setup
from training.standard_training import train
from training.her_training import her_train


if __name__ == '__main__':
    env, agent, video_file_name, EPISODES_NUM, EXPLORATORY_EPISODES_NUM, EPISODES_BETWEEN_SAVE = her_setup()
    her_train(env=env,
              agent=agent,
              video_file_name=video_file_name,
              total_epochs=EPISODES_NUM,
              EXPLORATORY_EPISODES_NUM=EXPLORATORY_EPISODES_NUM,
              EPISODES_BETWEEN_SAVE=EPISODES_BETWEEN_SAVE)


def plot_training_curve(episode_scores):
    import plotly.express as px
    fig = px.line(episode_scores, title='Training curve')
    fig.show()