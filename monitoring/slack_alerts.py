import os
import requests

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def send_slack_alert(kl_value, threshold, excess):
    if not SLACK_WEBHOOK_URL:
        print("Slack webhook URL not set")
        return

    message = {
        "text": (
            "Model Drift Detected!!\n"
            f"KL Divergence: `{kl_value:.4f}`\n"
            f"Threshold: `{threshold}`\n"
            f"Exceeded by: `{excess:.4f}`"
        )
    }

    requests.post(SLACK_WEBHOOK_URL, json=message)
