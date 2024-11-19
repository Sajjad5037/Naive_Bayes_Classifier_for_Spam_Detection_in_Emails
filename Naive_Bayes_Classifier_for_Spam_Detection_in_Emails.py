# 1. Import necessary libraries
import numpy as np

# 2. Define the training data (Spam and Ham words)
spam_words = ["free", "win", "money", "prize", "click", "offer", "urgent"]
ham_words = ["meeting", "schedule", "project", "team", "report", "discussion", "deadline"]

# 3. Function to train the Naive Bayes model
def train_naive_bayes(words, spam_words, ham_words):
    # Calculate prior probabilities
    P_spam = len(spam_words) / (len(spam_words) + len(ham_words))
    P_ham = len(ham_words) / (len(spam_words) + len(ham_words))
    
    # Calculate conditional probabilities with Laplace smoothing
    P_word_given_spam = {}
    P_word_given_ham = {}
    
    for word in words:
        P_word_given_spam[word] = (spam_words.count(word) + 1) / (len(spam_words) + len(words))
        P_word_given_ham[word] = (ham_words.count(word) + 1) / (len(ham_words) + len(words))
    
    return P_word_given_spam, P_word_given_ham, P_spam, P_ham

# Train the model
all_words = list(set(spam_words + ham_words))  # Unique vocabulary from both classes
P_word_given_spam, P_word_given_ham, P_spam, P_ham = train_naive_bayes(all_words, spam_words, ham_words)

# 4. Function to predict email class
def predict_email_class(email_words):
    P_email_spam = P_spam
    P_email_ham = P_ham
    
    for word in email_words:
        # Multiply probabilities for each word in the email
        P_email_spam *= P_word_given_spam.get(word, 1 / (len(spam_words) + len(all_words)))  # Laplace smoothing
        P_email_ham *= P_word_given_ham.get(word, 1 / (len(ham_words) + len(all_words)))  # Laplace smoothing
    
    # Compare probabilities and classify
    return "spam" if P_email_spam > P_email_ham else "ham"

# 5. Test the model with an example email
email_to_predict = ["free", "meeting", "prize"]
prediction = predict_email_class(email_to_predict)

# 6. Display the prediction result
print(f"The email is classified as: {prediction}")

# 7. Evaluate the model on more examples
test_emails = [
    (["win", "free", "money"], "Expected: spam"),
    (["meeting", "team", "project"], "Expected: ham"),
    (["urgent", "click", "offer"], "Expected: spam"),
    (["schedule", "report", "deadline"], "Expected: ham"),
]

print("\nModel Evaluation:")
for email, expected in test_emails:
    result = predict_email_class(email)
    print(f"Email: {email} -> Predicted: {result}, {expected}")
