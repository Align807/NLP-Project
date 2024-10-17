import streamlit as st
import pandas as pd
import torch
import pickle


with open('BERT.pkl', 'rb') as model_file:
    tokenizer, data, model, label_encoder = pickle.load(model_file)
# Predict function
def predict(text, tokenizer, model, label_encoder):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(predictions, dim=1).item()
    return label_encoder.inverse_transform([predicted_label])[0], predictions

# Main function for Streamlit app
def main():
    st.title("BERT Sentimental Analysis for user review")
    st.header("Enter Review of the Shoes for Prediction")
    user_input = st.text_area("Text Input", height=150, placeholder="Type your text here...")
    if st.button("Predict"):
            if user_input.strip():
                with st.spinner('Predicting...'):
                    try:
                        
                        predicted_label, predictions = predict(user_input, tokenizer, model, label_encoder)
                        st.write("Predicted Label:")
                        st.write(predicted_label)
                        st.write("Prediction Probabilities:")
                        prediction_table=pd.DataFrame(columns=["Negative", "Neutral","Positive"], data=predictions.numpy() )
                            
                        st.write(prediction_table)
                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
    else:
      st.warning("Please enter some text for prediction.")    
            
    

if __name__ == "__main__":
    main()
