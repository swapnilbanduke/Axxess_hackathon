from openai import OpenAI

# Initialize OpenAI client with your API key
client = OpenAI(api_key="your_openai_api_key_here")

# Example row data (replace with your actual data)
row = {
    'total_hcp_score': 85,
    'samples_shipped': 120,
    'talks_attended': 3,
    'completed_rx_written': 50,
    'predicted_gimoti_starts_decile': 9,
    'talks_signed_up': 2,
    'total_rx_written': 70,
    'days_since_last_call': 15
}

# Construct the prompt directly from row data
prompt = f"""
Generate an insight based on the following data:
- Total HCP Score: {row['total_hcp_score']}
- Samples Shipped: {row['samples_shipped']}
- Talks Attended: {row['talks_attended']}
- Completed Rx Written: {row['completed_rx_written']}
- Predicted Decile: {row['predicted_gimoti_starts_decile']}
- Talks Signed Up: {row['talks_signed_up']}
- Total Rx Written: {row['total_rx_written']}
- Days Since Last Call: {row['days_since_last_call']}

Explain why this Healthcare Professional (HCP) should be prioritized for outreach based on the provided data and priority score calculation.
"""

# Get insight from GPT
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=150,
    temperature=0.5
)

insight = response.choices[0].message.content.strip()

print("Generated Insight:", insight)
