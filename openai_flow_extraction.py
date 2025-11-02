"""
Use OpenAI API to extract flow and subflow from customer service transcripts.
Evaluates predictions against ground truth labels.
"""

import json
import pandas as pd
import os
from openai import OpenAI
from tqdm import tqdm
import time

# Configuration
DATA_PATH = "DATA/abcd_v1.1.json"
API_KEY = os.getenv("OPENAI_API_KEY")  # Set via: export OPENAI_API_KEY="your-key"
MODEL = "gpt-4o"  # Cheaper and fast alternative: "gpt-3.5-turbo"
OUTPUT_CSV = "results/flow_predictions.csv"

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)


def load_data(split="test"):
    """Load ABCD dataset and create DataFrame with transcripts and labels."""
    print(f"Loading {split} data...")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    rows = []
    for item in data[split]:
        # Create transcript
        transcript = " ".join([f"{speaker}: {text}"
                              for speaker, text in item["original"]])

        # Extract ground truth labels
        flow = item["scenario"].get("flow", "unknown")
        subflow = item["scenario"].get("subflow", "unknown")
        convo_id = item.get("convo_id", "")

        rows.append({
            "convo_id": convo_id,
            "transcript": transcript,
            "true_flow": flow,
            "true_subflow": subflow
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} conversations")
    return df


def get_unique_categories(df):
    """Get all unique flow and subflow categories from data."""
    flows = df["true_flow"].unique().tolist()
    subflows = df["true_subflow"].unique().tolist()
    return flows, subflows


def create_prompt(transcript, flows, subflows):
    """Create optimized prompt with targeted examples for common errors."""

    prompt = f"""E-commerce conversation classifier. Classify customer support conversations into precise flow categories.

## STEP-BY-STEP CLASSIFICATION LOGIC

### STEP 1: Check Order Status
**Does conversation mention "Order ID", "order number", or reference an existing order/purchase?**

**YES → Order ALREADY placed → Go to STEP 2**
**NO → No order yet OR general inquiry → Go to STEP 3**

---

### STEP 2: ORDER EXISTS - Determine the Issue Type

#### A. PRODUCT DEFECT - Physical product issues
**Use when:** Customer wants return/refund because item itself is wrong/bad
- Wrong item received (wrong size, color, style)
- Defective/damaged product
- Product quality issues
- "Changed my mind" / "Not what I expected" / "Don't want it anymore"
- **KEY:** Focus is on THE ITEM being unsatisfactory

**Examples:**
- "I want to return this jacket, it's not what I expected" → **product_defect**
- "The boots I received are damaged" → **product_defect**
- "Order ID 123. Wrong size, need to return" → **product_defect**

#### B. PURCHASE DISPUTE - Money/billing/payment issues
**Use when:** Problem is with CHARGES, PAYMENTS, or PRICING
- Charged wrong amount
- Charged twice / double charge
- Missing promo code discount (order placed without discount)
- Refund for a charge (focus on money, not product quality)
- Payment not processed correctly

**Examples:**
- "Order ID 456. You charged me twice!" → **purchase_dispute**
- "I used promo code but didn't get the discount" (order exists) → **purchase_dispute**
- "Order ID 789. Want refund for overcharge" → **purchase_dispute**

#### C. SHIPPING ISSUE - Delivery/tracking problems
**Use when:** Order placed, issue is with DELIVERY/SHIPMENT
- Package tracking
- Package lost/not received
- Late delivery
- Shipping address issues (after order placed)

**Examples:**
- "Where is my package? Order ID 321" → **shipping_issue**
- "Order hasn't arrived yet" → **shipping_issue**

---

### STEP 3: NO ORDER YET - Determine Activity Type

#### D. ORDER ISSUE - Trying to place order but encountering problems
**Use when:** Customer WANTS to order but something PREVENTS them
- Promo code won't work (before checkout)
- Can't complete purchase
- Item shows unavailable/out of stock (and they want to order)
- Cart issues preventing checkout
- **KEY:** Intention is to BUY, but blocked

**Examples:**
- "I'm trying to use promo code but it says invalid" (no order) → **order_issue**
- "Item I want is out of stock, can I backorder?" → **order_issue**
- "Can't complete my purchase" → **order_issue**

#### E. TROUBLESHOOT SITE - Technical/website problems
**Use when:** Website/technical errors (NOT about specific product)
- Credit card keeps declining
- Site won't load
- Login issues / can't access site
- Cart not working (technical glitch)
- **KEY:** TECHNICAL/SYSTEM problem

**Examples:**
- "My credit card keeps declining" → **troubleshoot_site**
- "Site won't load on my browser" → **troubleshoot_site**

#### F. ACCOUNT ACCESS - Login/password/authentication
**Use when:** Can't log in or access account
- Forgot password
- Account locked
- 2FA issues
- Username problems

**Examples:**
- "I forgot my password" → **account_access**
- "Can't log into my account" → **account_access**

#### G. MANAGE ACCOUNT - Updating account information
**Use when:** Customer wants to CHANGE account details (not subscription)
- Update shipping address (not for specific order)
- Change email/phone
- Update payment method on file
- Account settings changes
- **KEY:** Updating PROFILE/ACCOUNT info

**Examples:**
- "I moved, need to update my address on file" → **manage_account**
- "Want to change my email address" → **manage_account**

#### H. SUBSCRIPTION INQUIRY - Managing subscription
**Use when:** Customer HAS a subscription and wants to manage it
- Cancel subscription
- Change subscription
- Subscription billing issues
- **Must have EXISTING subscription**

**Examples:**
- "I have a subscription and want to cancel it" → **subscription_inquiry**
- "My subscription charged me twice" → **subscription_inquiry**

#### I. SINGLE ITEM QUERY - Questions about specific product
**Use when:** Asking about ONE particular product details
- Product specifications (size, color, material)
- Product availability check
- "Is this item in stock?"
- **KEY:** About a SPECIFIC item, not general policy

**Examples:**
- "What's the sleeve length of this shirt?" → **single_item_query**
- "Is this jacket available in size M?" → **single_item_query**

#### J. STOREWIDE QUERY - General policy/information questions
**Use when:** General questions about store/policies
- Return policy questions
- Shipping costs
- Promo code expiration (general question, no order)
- Store hours
- "How does X work?" (general)
- **KEY:** GENERAL info, not specific to one item or order

**Examples:**
- "What is your return policy?" → **storewide_query**
- "When do promo codes expire?" → **storewide_query**
- "How do I cancel a subscription?" (just asking, doesn't have one) → **storewide_query**

---

## CRITICAL DISTINCTIONS (Common Confusions)

**product_defect vs purchase_dispute:**
- Return because item is wrong/bad? → **product_defect**
- Problem with charges/payments? → **purchase_dispute**

**order_issue vs purchase_dispute:**
- No order yet, trying to buy? → **order_issue**
- Order exists, billing problem? → **purchase_dispute**

**manage_account vs subscription_inquiry:**
- Changing address/email/profile? → **manage_account**
- Managing subscription service? → **subscription_inquiry**

**single_item_query vs storewide_query:**
- Question about ONE product? → **single_item_query**
- General policy/how things work? → **storewide_query**

---

## CONVERSATION TO CLASSIFY:

{transcript}

---

## DECISION CHECKLIST:
1. ☐ Does an order already exist? (Order ID mentioned?)
2. ☐ If YES: Is it product quality, billing, or delivery issue?
3. ☐ If NO: Is it order attempt, question, or account management?
4. ☐ Double-check: Does my classification match the PRIMARY issue?

Valid flows: {', '.join(sorted(flows))}
Valid subflows: {', '.join(sorted(subflows))}

## RESPOND WITH JSON:
{{
  "flow": "<exact_flow_from_list_above>",
  "subflow": "<exact_subflow_from_list_above>"
}}"""

    return prompt


def extract_flow_subflow(transcript, flows, subflows, max_retries=3):
    """Call OpenAI API to extract flow and subflow from transcript."""
    prompt = create_prompt(transcript, flows, subflows)

    for attempt in range(max_retries):
        try:
            # Check if model supports JSON mode
            supports_json_mode = MODEL in ["gpt-4o", "gpt-3.5-turbo-1106", "gpt-4-1106-preview"]

            api_params = {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are an expert customer service issue classifier with deep understanding of e-commerce support conversations. You analyze conversations carefully and classify them into precise categories. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 200
            }

            # Only add response_format for models that support it
            if supports_json_mode:
                api_params["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**api_params)

            # Parse JSON response
            content = response.choices[0].message.content

            # Try to extract JSON from response (handles both pure JSON and text with JSON)
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the text
                import re
                json_match = re.search(r'\{[^{}]*"flow"[^{}]*"subflow"[^{}]*\}', content)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {content}")

            return result.get("flow", "unknown"), result.get("subflow", "unknown")

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry
            else:
                return "error", "error"


def process_batch(df, flows, subflows, limit=None):
    """Process all transcripts and get predictions."""
    print(f"\nProcessing transcripts with {MODEL}...")

    # Limit for testing (optional)
    if limit:
        df = df.head(limit)
        print(f"Processing only {limit} samples for testing")

    predictions = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        pred_flow, pred_subflow = extract_flow_subflow(
            row["transcript"],
            flows,
            subflows
        )

        predictions.append({
            "convo_id": row["convo_id"],
            "transcript": row["transcript"],
            "true_flow": row["true_flow"],
            "true_subflow": row["true_subflow"],
            "pred_flow": pred_flow,
            "pred_subflow": pred_subflow
        })

        # Rate limiting (OpenAI has limits)
        time.sleep(0.1)

    return pd.DataFrame(predictions)


def evaluate_predictions(results_df):
    """Calculate accuracy metrics."""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    # Flow accuracy
    flow_correct = (results_df["true_flow"] == results_df["pred_flow"]).sum()
    flow_accuracy = flow_correct / len(results_df) * 100

    # Subflow accuracy
    subflow_correct = (results_df["true_subflow"] == results_df["pred_subflow"]).sum()
    subflow_accuracy = subflow_correct / len(results_df) * 100

    # Both correct
    both_correct = ((results_df["true_flow"] == results_df["pred_flow"]) &
                    (results_df["true_subflow"] == results_df["pred_subflow"])).sum()
    both_accuracy = both_correct / len(results_df) * 100

    print(f"\nTotal samples: {len(results_df)}")
    print(f"\nFlow Accuracy: {flow_accuracy:.2f}% ({flow_correct}/{len(results_df)})")
    print(f"Subflow Accuracy: {subflow_accuracy:.2f}% ({subflow_correct}/{len(results_df)})")
    print(f"Both Correct: {both_accuracy:.2f}% ({both_correct}/{len(results_df)})")

    # Errors breakdown
    print("\n" + "-"*50)
    print("ERROR ANALYSIS")
    print("-"*50)

    flow_errors = results_df[results_df["true_flow"] != results_df["pred_flow"]]
    if len(flow_errors) > 0:
        print(f"\nFlow misclassifications: {len(flow_errors)}")
        print("\nMost common flow errors:")
        error_counts = flow_errors.groupby(["true_flow", "pred_flow"]).size().sort_values(ascending=False).head(5)
        for (true, pred), count in error_counts.items():
            print(f"  True: {true} → Predicted: {pred} ({count} times)")

    subflow_errors = results_df[results_df["true_subflow"] != results_df["pred_subflow"]]
    if len(subflow_errors) > 0:
        print(f"\nSubflow misclassifications: {len(subflow_errors)}")
        print("\nMost common subflow errors:")
        error_counts = subflow_errors.groupby(["true_subflow", "pred_subflow"]).size().sort_values(ascending=False).head(5)
        for (true, pred), count in error_counts.items():
            print(f"  True: {true} → Predicted: {pred} ({count} times)")

    return {
        "flow_accuracy": flow_accuracy,
        "subflow_accuracy": subflow_accuracy,
        "both_accuracy": both_accuracy
    }


def save_results(results_df):
    """Save results to CSV."""
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to: {OUTPUT_CSV}")


def main(split="test", limit=None):
    """Main pipeline: load data → predict → evaluate → save."""

    # Check API key
    if not API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    # Load data
    df = load_data(split)

    # Get valid categories (for prompt)
    flows, subflows = get_unique_categories(df)
    print(f"\nFound {len(flows)} flow categories and {len(subflows)} subflow categories")

    # Process transcripts
    results_df = process_batch(df, flows, subflows, limit=limit)

    # Evaluate
    metrics = evaluate_predictions(results_df)

    # Save results
    save_results(results_df)

    return results_df, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract flow/subflow using OpenAI API")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test",
                       help="Dataset split to process")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples (for testing)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                       help="OpenAI model to use")

    args = parser.parse_args()

    # Update model if specified
    MODEL = args.model

    print(f"Using model: {MODEL}")
    print(f"Processing split: {args.split}")

    results_df, metrics = main(split=args.split, limit=args.limit)

    # Print detailed accuracy summary
    print("\n" + "="*60)
    print("QUICK ACCURACY SUMMARY")
    print("="*60)
    print(f"Flow Accuracy:    {metrics['flow_accuracy']:6.2f}%")
    print(f"Subflow Accuracy: {metrics['subflow_accuracy']:6.2f}%")
    print(f"Both Correct:     {metrics['both_accuracy']:6.2f}%")
    print("="*60)
    print(f"\nFor detailed analysis, run: python3 analyze_accuracy.py")
    print("\n✅ Pipeline complete!")
