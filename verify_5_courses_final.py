"""Verify that generated CSV files have only 5 courses"""
import pandas as pd

print("🔍 Verifying Generated CSV Files...\n")

# Check ExpertSkillMap.csv
print("1️⃣ ExpertSkillMap.csv:")
expert = pd.read_csv('ExpertSkillMap.csv')
print(f"   Courses: {list(expert['Subject'].values)}")
print(f"   Count: {len(expert)} courses\n")

# Check if Target column exists in merged data
print("2️⃣ Checking data generation...")
cognitive = pd.read_csv('StudentCognitiveSkill.csv')
subject = pd.read_csv('StudentSubjectKnowledge.csv')
print(f"   Students in cognitive: {len(cognitive)}")
print(f"   Students in subject: {len(subject)}")
print(f"   Skills: {len([c for c in cognitive.columns if c.startswith('Skill')])}")
print(f"   Subjects: {len([c for c in subject.columns if c.startswith('Subject')])}\n")

print("="*60)
if len(expert) == 5:
    print("✅ SUCCESS! CSV files have exactly 5 courses:")
    for i, course in enumerate(expert['Subject'].values, 1):
        print(f"   {i}. {course}")
    print("\n📋 The app will show these 5 courses in recommendations.")
    print("   Open the app and go to 'Get Recommendation' page to see them.")
else:
    print(f"❌ ERROR: Found {len(expert)} courses instead of 5")
print("="*60)
