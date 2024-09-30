from moderatsiya_kartochek.main import make_submission

TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_DIR = "./data/"

if __name__ == "__main__":
    make_submission.callback(TEST_IMAGES_DIR, SUBMISSION_DIR)
