import os.path as path
import praw
import click
from praw.models import Comment, Submission
import secret
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(message)s', filename='data_read.log')
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('prawcore').setLevel(logging.CRITICAL)



@click.command()
@click.option('--directory', default='C:\\datasets', help='Directory to write to.')
@click.option('--limit', default=100, help='How many posts go through')
@click.option('--subreddit_names', '-s', multiple=True, help='The subreddit to fetch from')
def main(directory, subreddit_names, limit):
    for subreddit_name in subreddit_names:
        logging.info("Started fetching {0}".format(subreddit_name))
        fetch(directory, subreddit_name, limit)
        logging.info("Finished fetching {0}".format(subreddit_name))
    logging.info("Finished fetching all subreddits")


def fetch(directory, subreddit_name, limit):
    reddit = praw.Reddit(client_id=secret.client_id, client_secret=secret.client_secret, user_agent="redditMBTI")
    fullpath = path.join(directory, "{0}.tsv".format(subreddit_name))
    subreddit = reddit.subreddit(subreddit_name)
    with open(fullpath, "w", encoding="utf-8") as data_file:
        for post in subreddit.hot(limit=limit):
            try:
                write_post_data(data_file, post)
            except Exception as e:
                logging.error("error writing post {0}".format(post.permalink))

            for comment in post.comments:
                try:
                    write_comment_data(data_file, comment)
                except Exception as e:
                    logging.error("error writing comment {0}".format(comment.permalink))


def write_post_data(data_file, post):
    if isinstance(post, Submission):
        relevant_data = [
            str(post.score),
            post.title,
            post.selftext.replace('\n', '\\n').replace('\t', ' ')
        ]
        data_file.write('\t'.join(relevant_data) + '\n')


def write_comment_data(data_file, comment):
    if isinstance(comment, Comment):
        relevant_data = [
            str(comment.score),
            comment.body.replace('\n', '\\n').replace('\t', ' ')
        ]
        data_file.write('\t'.join(relevant_data) + '\n')


if __name__ == "__main__":
    main()
