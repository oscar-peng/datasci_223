# ... existing code ...

```python
# Create data directory if it doesn't exist
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

# Define the dataset directory name (using the actual name from the downloaded data)
dataset_dir = data_dir / 'a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0'

# Check if data exists and is in the correct structure
if (dataset_dir / 'Data').exists() and (dataset_dir / 'StudentGrades.txt').exists():
    print("Dataset already downloaded and in correct structure.")
    # Set data paths for later use
    data_path = dataset_dir / 'Data'
    grades_path = dataset_dir / 'StudentGrades.txt'
else:
    print("Downloading dataset from PhysioNet...")
    !wget -r -N -c -np https://physionet.org/files/wearable-exam-stress/1.0.0/ -P {data_dir}
    # Move files up from nested directory
    !mv {data_dir}/physionet.org/files/wearable-exam-stress/1.0.0/* {data_dir}/
    !rm -r {data_dir}/physionet.org
    print("Download complete!")
    # Set data paths after download
    data_path = dataset_dir / 'Data'
    grades_path = dataset_dir / 'StudentGrades.txt'

# Load grades
print("\nLoading student grades...")
grades_df = pd.read_csv(grades_path, sep='\t')
print("\nGrade Distribution Summary:")
print(grades_df.describe())

# Create an informative grade distribution plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=grades_df.melt(), x='variable', y='value', palette='viridis')
plt.title('Grade Distribution Across Exams')
plt.xlabel('Exam')
plt.ylabel('Grade (%)')

# Add individual points to show the actual distribution
sns.swarmplot(data=grades_df.melt(), x='variable', y='value', 
              color='red', alpha=0.5, size=4)

plt.show()

print("\n📊 Key Observations:")
print(f"- Average grade across all exams: {grades_df.mean().mean():.1f}%")
print(f"- Highest grade: {grades_df.max().max():.1f}%")
print(f"- Lowest grade: {grades_df.min().min():.1f}%")
print(f"- Number of students: {len(grades_df)}")

# ... existing code ... 