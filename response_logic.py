def process_response(gpt_response, sentiment):
    if sentiment["sentiment"] == "negative":
        return f"Thank you for sharing. It sounds like you’re going through a tough time. {gpt_response}"
    elif sentiment["sentiment"] == "positive":
        return f"That’s wonderful to hear! {gpt_response}"
    else:
        return gpt_response