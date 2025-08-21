import os
import json
import re
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI

class BillingAgent:
    def __init__(self):
        load_dotenv()
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2024-02-01"
        )

        # Fallback prices
        self.fallback_pricing = {
            "EC2": {"t3.micro": 0.0104, "t3.small": 0.0208, "t3.medium": 0.0416},
            "EKS": {"cluster": 0.10},
            "RDS": {"db.t3.micro": 0.017},
            "S3": {"storage": 0.023},
            "Lambda": {"request": 0.0000002}
        }

        # AWS pricing pages
        self.pricing_urls = {
                "EC2": "https://aws.amazon.com/ec2/pricing/on-demand/",
                "EKS": "https://aws.amazon.com/eks/pricing/",
                "RDS": "https://aws.amazon.com/rds/pricing/",
                "S3": "https://aws.amazon.com/s3/pricing/",
                "Lambda": "https://aws.amazon.com/lambda/pricing/",
                "ELB": "https://aws.amazon.com/elasticloadbalancing/pricing/",
                "CloudFront": "https://aws.amazon.com/cloudfront/pricing/",
                "DynamoDB": "https://aws.amazon.com/dynamodb/pricing/",
                "SQS": "https://aws.amazon.com/sqs/pricing/",
                "SNS": "https://aws.amazon.com/sns/pricing/"
            }
    def scrape_price(self, service, instance_type=None):
        """Scrape AWS pricing pages with regex fallback"""
        url = self.pricing_urls.get(service)
        if not url:
            return self._fallback(service, instance_type)

        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            html = requests.get(url, headers=headers, timeout=10).text.lower()

            # Look for instance-specific pricing
            if instance_type:
                patterns = [
                    rf"{instance_type.lower()}.*?\$(\d+\.?\d*)",
                    rf"\$(\d+\.?\d*).*?{instance_type.lower()}"
                ]
            else:
                patterns = [
                    r"\$(\d+\.\d+)\s*per\s*hour",
                    r"\$(\d+\.\d+)\s*hourly"
                ]

            for pattern in patterns:
                match = re.search(pattern, html)
                if match:
                    return float(match.group(1))
        except Exception as e:
            print(f"Scraping failed for {service}: {e}")

        return self._fallback(service, instance_type)

    def _fallback(self, service, instance_type):
        """Fallback to dictionary if scraping fails"""
        prices = self.fallback_pricing.get(service, {})
        if instance_type and instance_type in prices:
            return prices[instance_type]
        return list(prices.values())[0] if prices else 0.05

    def interpret_user_request(self, query):
        """LLM interprets AWS service + instance type"""
        system_prompt = """
            You are an AWS Pricing expert.
            A user may ask about a single service or a higher-level workload (like a Kubernetes cluster).
            Break it down into AWS services + instance types needed.

            Return ONLY valid JSON in this format:
            {
                "services": [
                    {
                    "service_code": "EC2",
                    "instance_type": "m5.large",
                    "count": 3,
                    "usage_estimate": "Worker nodes"
                    },
                    {
                    "service_code": "EKS",
                    "instance_type": null,
                    "count": 1,
                    "usage_estimate": "EKS control plane"
                    }
                ]
            }
            """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"The user asked: {query}"}
            ],
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(content)
        except:
            return {"service_code": "EC2", "instance_type": "t3.micro", "usage_estimate": "Default"}

    def estimate_cost(self, query):
        """Main flow: interpret → scrape → estimate (handles multiple services)"""
        parsed = self.interpret_user_request(query)
        services = parsed["services"]

        breakdown = []
        total_monthly = 0

        for svc in services:
            service = svc["service_code"]
            itype = svc.get("instance_type")
            count = svc.get("count", 1)

            price = self.scrape_price(service, itype)
            monthly = price * 730 * count
            yearly = monthly * 12
            total_monthly += monthly

            breakdown.append({
                "service": service,
                "instance_type": itype or "N/A",
                "count": count,
                "assumptions": svc["usage_estimate"],
                "hourly_price_usd": price,
                "monthly_estimate_usd": round(monthly, 2),
                "yearly_estimate_usd": round(yearly, 2)
            })

        return {
            "breakdown": breakdown,
            "total_monthly_usd": round(total_monthly, 2),
            "total_yearly_usd": round(total_monthly * 12, 2),
            "note": "💡 Includes multiple AWS services if needed"
        }

# Example usage
if __name__ == "__main__":
    agent = BillingAgent()
    query = "I would like to create a k8s cluster on AWS"
    print(json.dumps(agent.estimate_cost(query), indent=2))
