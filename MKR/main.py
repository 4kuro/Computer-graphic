import numpy as np  # Імпорт бібліотеки numpy для чисельних обчислень
import matplotlib.pyplot as plt  # Імпорт бібліотеки matplotlib для візуалізації

# Перевірка, чи число не містить заборонені цифри
def is_valid_number(num, forbidden_digits):
    num_str = str(num)  # Перетворюємо число у рядок
    return all(digit not in num_str for digit in forbidden_digits)  # Перевіряємо, що жодна заборонена цифра не входить до рядка

# Генерація точок фрактала
def generate_fractal_points(x_forbidden, y_forbidden, precision=4):
    points = []  # Список для зберігання точок фрактала
    for i in range(10**precision):  # Перебираємо всі можливі значення x з заданою точністю
        x = i / (10**precision)  # Розраховуємо значення x
        if not is_valid_number(i, x_forbidden):  # Перевіряємо, чи містить x заборонені цифри
            continue  # Пропускаємо, якщо містить
        for j in range(10**precision):  # Перебираємо всі можливі значення y з заданою точністю
            y = j / (10**precision)  # Розраховуємо значення y
            if is_valid_number(j, y_forbidden):  # Перевіряємо, чи не містить y заборонені цифри
                points.append((x, y))  # Додаємо точку (x, y) до списку, якщо умови виконуються
    return points  # Повертаємо список точок фрактала

# Параметри генерації
x_forbidden = ['2', '7']  # Заборонені цифри для x
y_forbidden = ['3', '5', '8']  # Заборонені цифри для y
precision = 4  # Точність обчислень (кількість знаків після коми)

points = generate_fractal_points(x_forbidden, y_forbidden, precision)  # Генеруємо точки фрактала

# Візуалізація фрактала
x, y = zip(*points)  # Розпаковуємо список точок у два списки: x і y
plt.figure(figsize=(8, 8))  # Створюємо нову фігуру для графіку з розмірами 8x8 дюймів
plt.scatter(x, y, s=0.1)  # Наносимо точки на графік з розміром маркерів 0.1
plt.title("Фрактал на площині")  # Додаємо заголовок графіку
plt.xlabel("x")  # Додаємо підпис до осі x
plt.ylabel("y")  # Додаємо підпис до осі y
plt.show()  # Відображаємо графік

# Обчислення кількості коробок для кожного розміру
def box_counting(points, sizes):
    counts = []  # Список для зберігання кількості коробок для кожного розміру
    for size in sizes:  # Перебираємо всі задані розміри коробок
        num_boxes = 0  # Лічильник для коробок, що містять частину фрактала
        for i in range(0, int(1 / size)):  # Перебираємо всі можливі позиції коробок по осі x
            for j in range(0, int(1 / size)):  # Перебираємо всі можливі позиції коробок по осі y
                for (x, y) in points:  # Перебираємо всі точки фрактала
                    if i * size <= x < (i + 1) * size and j * size <= y < (j + 1) * size:  # Перевіряємо, чи потрапляє точка у поточну коробку
                        num_boxes += 1  # Збільшуємо лічильник, якщо точка потрапляє у коробку
                        break  # Виходимо з внутрішнього циклу, якщо знайшли точку у поточній коробці
        counts.append(num_boxes)  # Додаємо кількість коробок для поточного розміру до списку
    return counts  # Повертаємо список кількості коробок

# Обчислення фрактальної розмірності
sizes = np.logspace(-1, -4, num=20)  # Генеруємо 20 логарифмічно рівномірно розподілених значень розмірів від 0.1 до 0.0001
counts = box_counting(points, sizes)  # Обчислюємо кількість коробок для кожного розміру

log_sizes = np.log(sizes)  # Обчислюємо логарифми розмірів коробок
log_counts = np.log(counts)  # Обчислюємо логарифми кількості коробок

# Побудова графіку для визначення фрактальної розмірності
plt.figure()  # Створюємо нову фігуру для графіку
plt.plot(log_sizes, log_counts, 'o-')  # Будуємо графік логарифмів розмірів проти логарифмів кількості коробок
plt.title("Метод коробок")  # Додаємо заголовок графіку
plt.xlabel("log(size)")  # Додаємо підпис до осі x
plt.ylabel("log(count)")  # Додаємо підпис до осі y
plt.show()  # Відображаємо графік

# Обчислення нахилу графіку (фрактальна розмірність)
fractal_dimension = -np.polyfit(log_sizes, log_counts, 1)[0]  # Обчислюємо нахил лінійної регресії на графіку, що є фрактальною розмірністю
print(f"Фрактальна розмірність: {fractal_dimension:.4f}")  # Виводимо результат
