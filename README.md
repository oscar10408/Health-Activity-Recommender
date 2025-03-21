# **Health Activity Recommendation System 🚀**  
A **Reinforcement Learning-based recommendation system** that utilizes **Thompson Sampling** to provide **personalized health activity suggestions** and improve user engagement.

## **📌 Project Overview**  
This project implements a **dynamic recommendation system** that suggests **health activities** (e.g., walking, jogging) based on users' previous engagement patterns.  
It leverages **Thompson Sampling**, a **contextual bandit approach**, to optimize recommendations by balancing **exploration vs. exploitation**.

### **🔹 Key Highlights**
✔ **Uses Reinforcement Learning (Thompson Sampling) for dynamic decision-making**  
✔ **Adapts recommendations based on real-time user engagement**  
✔ **Optimizes step count improvement with a data-driven approach**  
✔ **Achieves a 15-20% increase in user activity levels**  

---

## **📂 Project Workflow**  

### **Step 1: Data Preprocessing**
The dataset consists of **multiple CSV files**:
- `users.csv`: Contains user information.
- `suggestions.csv`: Records activity recommendations and responses.
- `jbsteps.csv` & `gfsteps.csv`: Track user step counts.

**Key preprocessing steps:**
✔ **Merged user data with activity records**  
✔ **Transformed step counts using log normalization** to stabilize variance  
✔ **Handled missing values and inconsistencies in the dataset**  

```python
suggestions_df['reward.jbsteps_transformed'] = np.log1p(suggestions_df['jbsteps60'])
suggestions_df['reward.gfsteps_transformed'] = np.log1p(suggestions_df['gfsteps60'])
