import os
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CLIAgent:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version="2024-02-01")

    def generate_cli_command(self, user_query: str) -> str:
        logger.info("Generating CLI command for user query: %s", user_query)

        system_prompt = """
        You are an AWS CLI command generator.
        - Take any user request related to AWS.
        - Identify the service, operation, and arguments.
        - Output ONLY the AWS CLI command.
        - If values are missing, use placeholders like <your-bucket-name> or <your-instance-id>.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0
        )

        return response.choices[0].message.content.strip()


# Example usage
if __name__ == "__main__":
    cli_agent = CLIAgent()
    while True:
        user_query = input("What AWS action do you want? (type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        command = cli_agent.generate_cli_command(user_query)
        #print(f"\nGenerated AWS CLI Command:\n{command}\n")
