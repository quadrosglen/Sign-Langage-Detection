import numpy as np
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Function to load the model and perform predictions
def perform_sign_language_prediction(model, keypoints):
    res = model.predict(np.expand_dims(keypoints, axis=0))[0]
    return res

# Function to visualize the output probabilities
def prob_viz(res, actions, colors):
    for num, prob in enumerate(res):
        st.sidebar.progress(prob, label=actions[num])
        col = st.sidebar.beta_columns(2)
        col[0].write(actions[num])
        col[1].write(f"{prob:.2f}")

# Main function for the Streamlit app
def main():
    mp_holistic = mp.solutions.holistic

    # Load the trained model
    model = load_model("action.h5")

    # Streamlit app configuration
    st.title("Sign Language Detection")
    st.sidebar.title("Action Probabilities")
    st.sidebar.text("Select the actions you want to recognize:")
    actions = np.array(["hello", "thanks", "please", "iloveyou", "takecare"])
    selected_actions = st.sidebar.multiselect("Actions", actions, default=actions)

    # Video capture using OpenCV
    cap = cv2.VideoCapture(0)

    # Apply Mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read camera feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            # Setup for predictions
            keypoints = extract_keypoints(results)

            if len(selected_actions) > 0:
                # Perform sign language prediction
                res = perform_sign_language_prediction(model, keypoints)

                # Visualize output probabilities
                prob_viz(res, actions, colors)

                # Predict sign language
                predicted_action_index = np.argmax(res)
                predicted_action = actions[predicted_action_index]

                # Display predicted action
                cv2.putText(
                    image, f"Predicted: {predicted_action}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                )

            # Display the webcam feed with detections and predictions
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="BGR")

            # Press 'q' to quit the app
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
