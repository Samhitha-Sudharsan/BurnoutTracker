from datetime import date, timedelta

def calculate_menstrual_phase(current_date: date, cycle_start_date: date, cycle_length: int = 28) -> dict:
    """
    Calculates the current menstrual phase based on the current date,
    cycle start date, and approximate cycle length.

    Args:
        current_date (date): The date for which to calculate the phase.
        cycle_start_date (date): The start date of the user's last menstrual period.
        cycle_length (int): The approximate length of the user's menstrual cycle in days.

    Returns:
        dict: A dictionary containing the 'phase' (str) and 'cycle_day' (int).
    """
    if cycle_length <= 0:
        raise ValueError("Cycle length must be a positive integer.")
    if cycle_start_date > current_date:
        return {"phase": "Not Started", "cycle_day": None}

    # Calculate days since the last period start
    # Use a loop to find the *most recent* cycle start before or on current_date
    # This handles cases where current_date is far after the initial cycle_start_date
    adjusted_cycle_start = cycle_start_date
    while adjusted_cycle_start + timedelta(days=cycle_length) <= current_date:
        adjusted_cycle_start += timedelta(days=cycle_length)

    days_since_start = (current_date - adjusted_cycle_start).days + 1 # +1 to make day 1 of cycle

    phase = ""
    is_pms = False

    if 1 <= days_since_start <= 5: # Assuming period is roughly 1-5 days
        phase = "Menstrual (Period)"
    elif 6 <= days_since_start <= 13: # Follicular phase
        phase = "Follicular"
    elif days_since_start == 14: # Ovulation day
        phase = "Ovulation"
    elif 15 <= days_since_start <= cycle_length: # Luteal phase
        phase = "Luteal"
        if days_since_start >= cycle_length - 5: # Last 5 days of Luteal can be PMS
            is_pms = True
    else:
        phase = "Irregular Cycle" # Should not happen with modulo, but as a fallback

    # Include PMS status in the phase name if relevant
    if is_pms:
        phase += " (PMS likely)"

    return {"phase": phase, "cycle_day": days_since_start}

# Example Usage (for testing the function directly)
if __name__ == "__main__":
    from datetime import date

    # Example 1: Today within a cycle
    today_date = date(2025, 6, 10) # Adjust to today's date if you want
    last_period_start = date(2025, 6, 1)
    cycle_len = 28
    result = calculate_menstrual_phase(today_date, last_period_start, cycle_len)
    print(f"For {today_date}, with cycle starting {last_period_start} and length {cycle_len}:")
    print(f"  Phase: {result['phase']}, Cycle Day: {result['cycle_day']}")

    # Example 2: Different day in the same cycle
    another_day = date(2025, 6, 15)
    result = calculate_menstrual_phase(another_day, last_period_start, cycle_len)
    print(f"For {another_day}, with cycle starting {last_period_start} and length {cycle_len}:")
    print(f"  Phase: {result['phase']}, Cycle Day: {result['cycle_day']}")

    # Example 3: Date beyond one cycle, should reference the next cycle start
    next_cycle_day = date(2025, 7, 5) # This would be day 5 of the *next* cycle
    result = calculate_menstrual_phase(next_cycle_day, last_period_start, cycle_len)
    print(f"For {next_cycle_day}, with cycle starting {last_period_start} and length {cycle_len}:")
    print(f"  Phase: {result['phase']}, Cycle Day: {result['cycle_day']}")

    # Example 4: Edge case - very short cycle
    short_cycle_day = date(2025, 6, 10)
    short_period_start = date(2025, 6, 8)
    short_cycle_len = 21
    result = calculate_menstrual_phase(short_cycle_day, short_period_start, short_cycle_len)
    print(f"For {short_cycle_day}, with cycle starting {short_period_start} and length {short_cycle_len}:")
    print(f"  Phase: {result['phase']}, Cycle Day: {result['cycle_day']}")