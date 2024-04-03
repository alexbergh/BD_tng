def unique_values(lst):
    unique = []
    for number in lst:
        if lst.count(number) == 1:
            unique.append(number)
    return unique

print(unique_values([1, 2, 2, 3, 4, 4, 5]))  # Пример использования функции
