def reverse_list(lst):
    # Возвращаем список в обратном порядке
    return lst[::-1]

# Исходный список
initial_list = ["male", "male", "female", "male", "male", "female", "female"]

# Переворачиваем список
new_list = reverse_list(initial_list)

def count_genders(lst):
    male_count = lst.count("male")
    female_count = lst.count("female")
    print(f"Кол-во мужчин: {male_count}, Кол-во женщин: {female_count}")

# Вызов функции count_genders с новым списком
count_genders(new_list)
