import pandas as pd
import numpy as np
import networkx as nx
import shap
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

# ==========================================
# 1. DIAGNOSTIC ENGINE (ML PIPELINE)
# ==========================================

class DiagnosticEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.model = None
        self.le_gender = LabelEncoder()
        self.le_stream = LabelEncoder()
        self.explainer = None
        self._train_model()

    def _train_model(self):
        self.df['Gender'] = self.le_gender.fit_transform(self.df['Gender'])
        self.df['Stream'] = self.le_stream.fit_transform(self.df['Stream'])
        
        X = self.df.drop('PlacedOrNot', axis=1)
        y = self.df['PlacedOrNot']
        
        self.model = XGBClassifier(eval_metric='logloss')
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def get_streams(self):
        return list(self.le_stream.classes_)

    def analyze(self, age, gender, stream, internships, cgpa, hostel, backlogs):
        try:
            gender_code = self.le_gender.transform([gender])[0]
            stream_code = self.le_stream.transform([stream])[0]
            
            input_vector = np.array([[age, gender_code, stream_code, internships, cgpa, hostel, backlogs]])
            
            # Predict probability
            prob = self.model.predict_proba(input_vector)[0][1]
            
            # Get SHAP impacts
            shap_values = self.explainer.shap_values(input_vector)
            feature_names = ['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'Backlogs']
            impact_tuples = sorted(zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)
            
            return prob, impact_tuples
        except Exception as e:
            print(f"Analysis Error: {e}")
            return 0.0, []

# ==========================================
# 2. ROUTING ENGINE (GRAPH PIPELINE)
# ==========================================

class RoutingEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.strip()
        self.G = nx.Graph()
        self.all_unique_skills = set()
        self.all_jobs = set()
        self._build_graph()

    def _build_graph(self):
        self.df['All_Skills'] = (
            self.df['Skills'].fillna('') + "," +
            self.df.get('Programming Languages', pd.Series([''] * len(self.df))) + "," +
            self.df['Tools'].fillna('')
        )
        df_exploded = self.df.assign(All_Skills=self.df['All_Skills'].str.split(',')).explode('All_Skills')
        df_exploded['All_Skills'] = df_exploded['All_Skills'].str.strip()
        df_exploded = df_exploded[df_exploded['All_Skills'] != '']

        for _, row in df_exploded.iterrows():
            job = row['Job roles']
            skill = row['All_Skills']
            if isinstance(skill, str) and len(skill) > 1:
                self.G.add_edge(job, skill)
                self.all_unique_skills.add(skill)
                self.all_jobs.add(job)

    def get_skill_list(self):
        return sorted([str(s) for s in self.all_unique_skills if isinstance(s, str)])

    def get_job_list(self):
        return sorted(list(self.all_jobs))

    def get_gap(self, target_job, user_skills):
        if target_job not in self.G.nodes:
            return []
        required = set(self.G.neighbors(target_job))
        current = set(user_skills)
        missing = list(required - current)
        return missing

    def recommend(self, user_skills):
        possible_jobs = []
        for skill in user_skills:
            if skill in self.G.nodes:
                neighbors = list(self.G.neighbors(skill))
                possible_jobs.extend(neighbors)

        if not possible_jobs:
            return None, []

        top_jobs_counter = Counter(possible_jobs).most_common(1)
        best_job = top_jobs_counter[0][0]
        missing = self.get_gap(best_job, user_skills)

        return best_job, missing

# ==========================================
# 3. EXECUTION SCRIPT
# ==========================================

def main():
    # 1. File Locator Logic
    college_files = ['collegePlace.csv', 'college_place.csv']
    career_files = ['Tech_Data_Cleaned.csv', 'career path dataset.csv', 'career_path.csv']

    def find_file(filename_options):
        search_paths = ['.', '/content', os.getcwd()]
        for path in search_paths:
            for fname in filename_options:
                full_path = os.path.join(path, fname)
                if os.path.exists(full_path):
                    return full_path
        return None

    place_path = find_file(college_files)
    career_path = find_file(career_files)

    if not place_path or not career_path:
        print("❌ CSV Files not found. Please ensure your datasets are in the directory.")
        return

    print("✅ Datasets found. Initializing engines...")
    diag_engine = DiagnosticEngine(place_path)
    route_engine = RoutingEngine(career_path)

    # 2. Hardcoded Input Variables (Replace these with your dynamic inputs)
    user_input = {
        "age": 21,
        "gender": "Male",
        "stream": "Computer Science",
        "internships": 2,
        "cgpa": 8.5,
        "hostel": True,
        "backlogs": False,
        "skills": ["Python", "SQL", "Machine Learning"],
        "desired_role": "Data Scientist"
    }

    print("\n" + "="*40)
    print("🎓 STUDENT PROFILE")
    print("="*40)
    for k, v in user_input.items():
        print(f"{k.capitalize()}: {v}")

    # 3. Run Pipeline
    # Convert booleans to integers for the ML model
    h_val = 1 if user_input["hostel"] else 0
    b_val = 1 if user_input["backlogs"] else 0

    # ML Analysis
    prob, impacts = diag_engine.analyze(
        user_input["age"], user_input["gender"], user_input["stream"], 
        user_input["internships"], user_input["cgpa"], h_val, b_val
    )

    # Graph Routing
    rec_job, rec_missing = route_engine.recommend(user_input["skills"])
    dream_missing = route_engine.get_gap(user_input["desired_role"], user_input["skills"]) if user_input["desired_role"] else []

    # 4. Print Results
    print("\n" + "="*40)
    print("📊 DIAGNOSTIC RESULTS")
    print("="*40)
    print(f"Placement Probability: {prob * 100:.2f}%")
    print("Key Driving Factors (SHAP):")
    for feature, val in impacts[:4]:
        sign = "+" if val > 0 else ""
        print(f"  • {feature}: {sign}{val:.3f}")

    print("\n" + "="*40)
    print("🧭 ROUTING & GAP ANALYSIS")
    print("="*40)
    
    print("AI RECOMMENDED ROLE (Easiest Path based on current skills):")
    if rec_job:
        print(f"  • Target: {rec_job}")
        print(f"  • Missing Skills: {', '.join(rec_missing) if rec_missing else 'None - Ready!'}")
    else:
        print("  • Target: None found based on current skills.")

    if user_input["desired_role"]:
        print("\nDESIRED ROLE (Dream Job):")
        print(f"  • Target: {user_input['desired_role']}")
        print(f"  • Missing Skills: {', '.join(dream_missing) if dream_missing else 'None - Ready!'}")

if __name__ == "__main__":

    main()


# another example:



import pandas as pd
import numpy as np
import networkx as nx
import shap
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

# ==========================================
# 1. ML PIPELINE (Diagnostic Engine)
# ==========================================

class DiagnosticEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.model = None
        self.le_gender = LabelEncoder()
        self.le_stream = LabelEncoder()
        self.explainer = None
        self._train_model()

    def _train_model(self):
        self.df['Gender'] = self.le_gender.fit_transform(self.df['Gender'])
        self.df['Stream'] = self.le_stream.fit_transform(self.df['Stream'])
        X = self.df.drop('PlacedOrNot', axis=1)
        y = self.df['PlacedOrNot']
        self.model = XGBClassifier(eval_metric='logloss')
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def get_streams(self):
        return list(self.le_stream.classes_)

    def analyze(self, age, gender, stream, internships, cgpa, hostel, backlogs):
        try:
            gender_code = self.le_gender.transform([gender])[0]
            stream_code = self.le_stream.transform([stream])[0]
            input_vector = np.array([[age, gender_code, stream_code, internships, cgpa, hostel, backlogs]])
            prob = self.model.predict_proba(input_vector)[0][1]
            shap_values = self.explainer.shap_values(input_vector)
            feature_names = ['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'Backlogs']
            impact_tuples = sorted(zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)
            return prob, impact_tuples
        except Exception as e:
            print(f"Error in ML analysis: {e}")
            return 0.0, []

# ==========================================
# 2. GRAPH PIPELINE (Routing Engine)
# ==========================================

class RoutingEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.strip()
        self.G = nx.Graph()
        self.all_unique_skills = set()
        self.all_jobs = set()
        self._build_graph()

    def _build_graph(self):
        self.df['All_Skills'] = (
            self.df['Skills'].fillna('') + "," +
            self.df.get('Programming Languages', pd.Series([''] * len(self.df))) + "," +
            self.df['Tools'].fillna('')
        )
        df_exploded = self.df.assign(All_Skills=self.df['All_Skills'].str.split(',')).explode('All_Skills')
        df_exploded['All_Skills'] = df_exploded['All_Skills'].str.strip()
        df_exploded = df_exploded[df_exploded['All_Skills'] != '']

        for _, row in df_exploded.iterrows():
            job = row['Job roles']
            skill = row['All_Skills']
            if isinstance(skill, str) and len(skill) > 1:
                self.G.add_edge(job, skill)
                self.all_unique_skills.add(skill)
                self.all_jobs.add(job)

    def get_skill_list(self):
        clean_list = [str(s) for s in self.all_unique_skills if isinstance(s, str)]
        return sorted(clean_list)

    def get_job_list(self):
        return sorted(list(self.all_jobs))

    def get_gap(self, target_job, user_skills):
        if target_job not in self.G.nodes:
            return []
        required = set(self.G.neighbors(target_job))
        current = set(user_skills)
        missing = list(required - current)
        return missing

    def recommend(self, user_skills):
        possible_jobs = []
        for skill in user_skills:
            if skill in self.G.nodes:
                neighbors = list(self.G.neighbors(skill))
                possible_jobs.extend(neighbors)

        if not possible_jobs:
            return None, []

        top_jobs_counter = Counter(possible_jobs).most_common(1)
        best_job = top_jobs_counter[0][0]
        missing = self.get_gap(best_job, user_skills)

        return best_job, missing

# ==========================================
# 3. EXECUTION & OUTPUT
# ==========================================

def find_file(filename_options):
    search_paths = ['.', '/content', os.getcwd()]
    for path in search_paths:
        for fname in filename_options:
            full_path = os.path.join(path, fname)
            if os.path.exists(full_path):
                return full_path
    return None

if __name__ == "__main__":
    print("--- Loading Data ---")
    career_files = ['Tech_Data_Cleaned.csv', 'career path dataset.csv', 'career_path.csv']
    college_files = ['collegePlace.csv', 'college_place.csv']

    place_path = find_file(college_files)
    career_path = find_file(career_files)

    if not place_path or not career_path:
        print("❌ ERROR: CSV Files not found. Please ensure 'collegePlace.csv' and 'career path dataset.csv' are in the directory.")
    else:
        print("✅ Data found. Initializing engines...")
        diag_engine = DiagnosticEngine(place_path)
        route_engine = RoutingEngine(career_path)

        # --- SAMPLE INPUT DATA ---
        student_profile = {
            "age": 21,
            "gender": "Male",
            "stream": "Computer Science",
            "internships": 1,
            "cgpa": 7.5,
            "hostel": 1, # 1 for Yes, 0 for No
            "backlogs": 0 # 1 for Yes, 0 for No
        }
        
        user_skills = ["Python", "Machine Learning", "SQL"]
        desired_role = "Data Scientist"

        print("\n" + "="*40)
        print("📊 PLACEMENT PREDICTION ANALYSIS")
        print("="*40)
        
        prob, impacts = diag_engine.analyze(**student_profile)
        
        print(f"Placement Probability: {prob * 100:.2f}%\n")
        print("Key Influencers (SHAP Values):")
        for feature, val in impacts:
            sign = "+" if val > 0 else ""
            print(f"  - {feature}: {sign}{val:.4f}")

        print("\n" + "="*40)
        print("🧭 SKILL GAP & ROUTING ANALYSIS")
        print("="*40)

        # AI Recommendation Path
        rec_job, rec_missing = route_engine.recommend(user_skills)
        print(f"AI Recommended Role (Easiest Path): {rec_job}")
        if not rec_missing:
            print("  ✅ You have all the required skills for this role!")
        else:
            print(f"  ❌ Missing Skills ({len(rec_missing)}): {', '.join(rec_missing)}")

        print("-" * 40)

        # Desired Role Path
        dream_missing = route_engine.get_gap(desired_role, user_skills)
        print(f"Desired Role (Ambitious Path): {desired_role}")
        if not dream_missing:
            print("  ✅ You have all the required skills for your dream role!")
        else:
            print(f"  ❌ Missing Skills ({len(dream_missing)}): {', '.join(dream_missing)}")
        
        print("\nDone.")


import pandas as pd
import numpy as np
import networkx as nx
import shap
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

# ==========================================
# 1. DIAGNOSTIC ENGINE (ML PIPELINE)
# ==========================================

class DiagnosticEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.model = None
        self.le_gender = LabelEncoder()
        self.le_stream = LabelEncoder()
        self.explainer = None
        self._train_model()

    def _train_model(self):
        self.df['Gender'] = self.le_gender.fit_transform(self.df['Gender'])
        self.df['Stream'] = self.le_stream.fit_transform(self.df['Stream'])
        
        X = self.df.drop('PlacedOrNot', axis=1)
        y = self.df['PlacedOrNot']
        
        self.model = XGBClassifier(eval_metric='logloss')
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def get_streams(self):
        return list(self.le_stream.classes_)

    def analyze(self, age, gender, stream, internships, cgpa, hostel, backlogs):
        try:
            gender_code = self.le_gender.transform([gender])[0]
            stream_code = self.le_stream.transform([stream])[0]
            
            input_vector = np.array([[age, gender_code, stream_code, internships, cgpa, hostel, backlogs]])
            
            # Predict probability
            prob = self.model.predict_proba(input_vector)[0][1]
            
            # Get SHAP impacts
            shap_values = self.explainer.shap_values(input_vector)
            feature_names = ['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'Backlogs']
            impact_tuples = sorted(zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)
            
            return prob, impact_tuples
        except Exception as e:
            print(f"Analysis Error: {e}")
            return 0.0, []

# ==========================================
# 2. ROUTING ENGINE (GRAPH PIPELINE)
# ==========================================

class RoutingEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.strip()
        self.G = nx.Graph()
        self.all_unique_skills = set()
        self.all_jobs = set()
        self._build_graph()

    def _build_graph(self):
        self.df['All_Skills'] = (
            self.df['Skills'].fillna('') + "," +
            self.df.get('Programming Languages', pd.Series([''] * len(self.df))) + "," +
            self.df['Tools'].fillna('')
        )
        df_exploded = self.df.assign(All_Skills=self.df['All_Skills'].str.split(',')).explode('All_Skills')
        df_exploded['All_Skills'] = df_exploded['All_Skills'].str.strip()
        df_exploded = df_exploded[df_exploded['All_Skills'] != '']

        for _, row in df_exploded.iterrows():
            job = row['Job roles']
            skill = row['All_Skills']
            if isinstance(skill, str) and len(skill) > 1:
                self.G.add_edge(job, skill)
                self.all_unique_skills.add(skill)
                self.all_jobs.add(job)

    def get_skill_list(self):
        return sorted([str(s) for s in self.all_unique_skills if isinstance(s, str)])

    def get_job_list(self):
        return sorted(list(self.all_jobs))

    def get_gap(self, target_job, user_skills):
        if target_job not in self.G.nodes:
            return []
        required = set(self.G.neighbors(target_job))
        current = set(user_skills)
        missing = list(required - current)
        return missing

    def recommend(self, user_skills):
        possible_jobs = []
        for skill in user_skills:
            if skill in self.G.nodes:
                neighbors = list(self.G.neighbors(skill))
                possible_jobs.extend(neighbors)

        if not possible_jobs:
            return None, []

        top_jobs_counter = Counter(possible_jobs).most_common(1)
        best_job = top_jobs_counter[0][0]
        missing = self.get_gap(best_job, user_skills)

        return best_job, missing

# ==========================================
# 3. EXECUTION SCRIPT
# ==========================================

def main():
    # 1. File Locator Logic
    college_files = ['collegePlace.csv', 'college_place.csv']
    career_files = ['Tech_Data_Cleaned.csv', 'career path dataset.csv', 'career_path.csv']

    def find_file(filename_options):
        search_paths = ['.', '/content', os.getcwd()]
        for path in search_paths:
            for fname in filename_options:
                full_path = os.path.join(path, fname)
                if os.path.exists(full_path):
                    return full_path
        return None

    place_path = find_file(college_files)
    career_path = find_file(career_files)

    if not place_path or not career_path:
        print("❌ CSV Files not found. Please ensure your datasets are in the directory.")
        return

    print("✅ Datasets found. Initializing engines...")
    diag_engine = DiagnosticEngine(place_path)
    route_engine = RoutingEngine(career_path)

    # 2. Hardcoded Input Variables (Replace these with your dynamic inputs)
    user_input = {
        "age": 21,
        "gender": "Male",
        "stream": "Computer Science",
        "internships": 2,
        "cgpa": 8.5,
        "hostel": True,
        "backlogs": False,
        "skills": ["Python", "SQL", "Machine Learning"],
        "desired_role": "Data Scientist"
    }

    print("\n" + "="*40)
    print("🎓 STUDENT PROFILE")
    print("="*40)
    for k, v in user_input.items():
        print(f"{k.capitalize()}: {v}")

    # 3. Run Pipeline
    # Convert booleans to integers for the ML model
    h_val = 1 if user_input["hostel"] else 0
    b_val = 1 if user_input["backlogs"] else 0

    # ML Analysis
    prob, impacts = diag_engine.analyze(
        user_input["age"], user_input["gender"], user_input["stream"], 
        user_input["internships"], user_input["cgpa"], h_val, b_val
    )

    # Graph Routing
    rec_job, rec_missing = route_engine.recommend(user_input["skills"])
    dream_missing = route_engine.get_gap(user_input["desired_role"], user_input["skills"]) if user_input["desired_role"] else []

    # 4. Print Results
    print("\n" + "="*40)
    print("📊 DIAGNOSTIC RESULTS")
    print("="*40)
    print(f"Placement Probability: {prob * 100:.2f}%")
    print("Key Driving Factors (SHAP):")
    for feature, val in impacts[:4]:
        sign = "+" if val > 0 else ""
        print(f"  • {feature}: {sign}{val:.3f}")

    print("\n" + "="*40)
    print("🧭 ROUTING & GAP ANALYSIS")
    print("="*40)
    
    print("AI RECOMMENDED ROLE (Easiest Path based on current skills):")
    if rec_job:
        print(f"  • Target: {rec_job}")
        print(f"  • Missing Skills: {', '.join(rec_missing) if rec_missing else 'None - Ready!'}")
    else:
        print("  • Target: None found based on current skills.")

    if user_input["desired_role"]:
        print("\nDESIRED ROLE (Dream Job):")
        print(f"  • Target: {user_input['desired_role']}")
        print(f"  • Missing Skills: {', '.join(dream_missing) if dream_missing else 'None - Ready!'}")

if __name__ == "__main__":

    main()


# another example:



import pandas as pd
import numpy as np
import networkx as nx
import shap
import os
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from collections import Counter

# ==========================================
# 1. ML PIPELINE (Diagnostic Engine)
# ==========================================

class DiagnosticEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.model = None
        self.le_gender = LabelEncoder()
        self.le_stream = LabelEncoder()
        self.explainer = None
        self._train_model()

    def _train_model(self):
        self.df['Gender'] = self.le_gender.fit_transform(self.df['Gender'])
        self.df['Stream'] = self.le_stream.fit_transform(self.df['Stream'])
        X = self.df.drop('PlacedOrNot', axis=1)
        y = self.df['PlacedOrNot']
        self.model = XGBClassifier(eval_metric='logloss')
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)

    def get_streams(self):
        return list(self.le_stream.classes_)

    def analyze(self, age, gender, stream, internships, cgpa, hostel, backlogs):
        try:
            gender_code = self.le_gender.transform([gender])[0]
            stream_code = self.le_stream.transform([stream])[0]
            input_vector = np.array([[age, gender_code, stream_code, internships, cgpa, hostel, backlogs]])
            prob = self.model.predict_proba(input_vector)[0][1]
            shap_values = self.explainer.shap_values(input_vector)
            feature_names = ['Age', 'Gender', 'Stream', 'Internships', 'CGPA', 'Hostel', 'Backlogs']
            impact_tuples = sorted(zip(feature_names, shap_values[0]), key=lambda x: abs(x[1]), reverse=True)
            return prob, impact_tuples
        except Exception as e:
            print(f"Error in ML analysis: {e}")
            return 0.0, []

# ==========================================
# 2. GRAPH PIPELINE (Routing Engine)
# ==========================================

class RoutingEngine:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.df.columns = self.df.columns.str.strip()
        self.G = nx.Graph()
        self.all_unique_skills = set()
        self.all_jobs = set()
        self._build_graph()

    def _build_graph(self):
        self.df['All_Skills'] = (
            self.df['Skills'].fillna('') + "," +
            self.df.get('Programming Languages', pd.Series([''] * len(self.df))) + "," +
            self.df['Tools'].fillna('')
        )
        df_exploded = self.df.assign(All_Skills=self.df['All_Skills'].str.split(',')).explode('All_Skills')
        df_exploded['All_Skills'] = df_exploded['All_Skills'].str.strip()
        df_exploded = df_exploded[df_exploded['All_Skills'] != '']

        for _, row in df_exploded.iterrows():
            job = row['Job roles']
            skill = row['All_Skills']
            if isinstance(skill, str) and len(skill) > 1:
                self.G.add_edge(job, skill)
                self.all_unique_skills.add(skill)
                self.all_jobs.add(job)

    def get_skill_list(self):
        clean_list = [str(s) for s in self.all_unique_skills if isinstance(s, str)]
        return sorted(clean_list)

    def get_job_list(self):
        return sorted(list(self.all_jobs))

    def get_gap(self, target_job, user_skills):
        if target_job not in self.G.nodes:
            return []
        required = set(self.G.neighbors(target_job))
        current = set(user_skills)
        missing = list(required - current)
        return missing

    def recommend(self, user_skills):
        possible_jobs = []
        for skill in user_skills:
            if skill in self.G.nodes:
                neighbors = list(self.G.neighbors(skill))
                possible_jobs.extend(neighbors)

        if not possible_jobs:
            return None, []

        top_jobs_counter = Counter(possible_jobs).most_common(1)
        best_job = top_jobs_counter[0][0]
        missing = self.get_gap(best_job, user_skills)

        return best_job, missing

# ==========================================
# 3. EXECUTION & OUTPUT
# ==========================================

def find_file(filename_options):
    search_paths = ['.', '/content', os.getcwd()]
    for path in search_paths:
        for fname in filename_options:
            full_path = os.path.join(path, fname)
            if os.path.exists(full_path):
                return full_path
    return None

if __name__ == "__main__":
    print("--- Loading Data ---")
    career_files = ['Tech_Data_Cleaned.csv', 'career path dataset.csv', 'career_path.csv']
    college_files = ['collegePlace.csv', 'college_place.csv']

    place_path = find_file(college_files)
    career_path = find_file(career_files)

    if not place_path or not career_path:
        print("❌ ERROR: CSV Files not found. Please ensure 'collegePlace.csv' and 'career path dataset.csv' are in the directory.")
    else:
        print("✅ Data found. Initializing engines...")
        diag_engine = DiagnosticEngine(place_path)
        route_engine = RoutingEngine(career_path)

        # --- SAMPLE INPUT DATA ---
        student_profile = {
            "age": 21,
            "gender": "Male",
            "stream": "Computer Science",
            "internships": 1,
            "cgpa": 7.5,
            "hostel": 1, # 1 for Yes, 0 for No
            "backlogs": 0 # 1 for Yes, 0 for No
        }
        
        user_skills = ["Python", "Machine Learning", "SQL"]
        desired_role = "Data Scientist"

        print("\n" + "="*40)
        print("📊 PLACEMENT PREDICTION ANALYSIS")
        print("="*40)
        
        prob, impacts = diag_engine.analyze(**student_profile)
        
        print(f"Placement Probability: {prob * 100:.2f}%\n")
        print("Key Influencers (SHAP Values):")
        for feature, val in impacts:
            sign = "+" if val > 0 else ""
            print(f"  - {feature}: {sign}{val:.4f}")

        print("\n" + "="*40)
        print("🧭 SKILL GAP & ROUTING ANALYSIS")
        print("="*40)

        # AI Recommendation Path
        rec_job, rec_missing = route_engine.recommend(user_skills)
        print(f"AI Recommended Role (Easiest Path): {rec_job}")
        if not rec_missing:
            print("  ✅ You have all the required skills for this role!")
        else:
            print(f"  ❌ Missing Skills ({len(rec_missing)}): {', '.join(rec_missing)}")

        print("-" * 40)

        # Desired Role Path
        dream_missing = route_engine.get_gap(desired_role, user_skills)
        print(f"Desired Role (Ambitious Path): {desired_role}")
        if not dream_missing:
            print("  ✅ You have all the required skills for your dream role!")
        else:
            print(f"  ❌ Missing Skills ({len(dream_missing)}): {', '.join(dream_missing)}")
        
        print("\nDone.")