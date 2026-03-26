"""Force delete old CSV files and regenerate with 5 courses"""
import os
import time

# Try to delete old CSV files
files_to_delete = [
    'StudentCognitiveSkill.csv',
    'StudentSubjectKnowledge.csv', 
    'ExpertSkillMap.csv'
]

print("🗑️  Attempting to delete old CSV files...")
for file in files_to_delete:
    if os.path.exists(file):
        try:
            os.remove(file)
            print(f"   ✅ Deleted {file}")
        except Exception as e:
            print(f"   ❌ Could not delete {file}: {e}")
            print(f"      Please close Excel/CSV viewer and try again")
    else:
        print(f"   ℹ️  {file} already deleted")

print("\n" + "="*60)
print("After deleting the files, run: streamlit run app.py")
print("The app will generate new CSV files with 5 courses only.")
print("="*60)
