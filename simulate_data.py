import random
import pandas as pd

def random_choice(options):
    return random.choice(options)

def generate_fake_response():
    return {
        "hours_worked": random.randint(0, 16),
        "meetings_attended": random.randint(0, 10),
        "sleep": random.randint(4, 10),  # Integer sleep hours
        "caffeine_intake": random_choice(["None", "1 cup", "2 cups", "3+ cups"]),
        "meals_skipped": random_choice(["None", "1", "2", "All"]),
        "exercise_minutes": random.randint(0, 120),
        "screen_time": random.randint(1, 15),
        "stress_level": random_choice(["Low", "Medium", "High", "Very High"]),
        "mood": random_choice(["Happy", "Neutral", "Tired", "Frustrated"]),
        "water": random_choice(["<1L", "1-1.5L", "1.5-2.5L", "2.5L+"]),
        "social_interaction": random_choice(["None", "Little", "Moderate", "A lot"]),
        "tasks_completed": random.randint(0, 20),
        "breaks_taken": random.randint(0, 10),
        "burnout_rating": random.randint(1, 10),
        "willingness_to_work_next_day": random_choice(["Yes", "No", "Maybe"]),
        "motivation_level": random_choice(["Low", "Medium", "High"]),
        "distractions": random_choice(["Few", "Moderate", "Many"]),
    }

# Generate a dataset of 500 entries
data = [generate_fake_response() for _ in range(500)]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("simulated_burnout_data.csv", index=False)
