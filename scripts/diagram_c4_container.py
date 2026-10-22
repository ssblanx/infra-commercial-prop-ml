import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(17, 10.5))
ax.set_xlim(0, 18); ax.set_ylim(0, 11)
ax.axis('off')
ax.set_title('C4: Диаграмма контейнеров', fontsize=15, weight='bold', pad=14)

def box(x, y, w, h, text, color, text_color='white', fontsize=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                        linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
             color=text_color, fontsize=fontsize, zorder=4)

def arrow(x1, y1, x2, y2, label, lx, ly, fontsize=8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                          mutation_scale=14, color='#555555', linewidth=1.2, zorder=2)
    ax.add_patch(a)
    if label:
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fontsize,
                 color='#333333', zorder=5,
                 bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

# граница системы
boundary = Rectangle((4.0, 1.9), 13.6, 7.0, fill=False, linestyle='--',
                       edgecolor='#999999', linewidth=1.3, zorder=1)
ax.add_patch(boundary)
ax.text(4.2, 8.65, 'Система анализа влияния инфраструктуры на стоимость недвижимости',
         fontsize=9.5, color='#777777', style='italic')

# внешние акторы
box(0.3, 8.0, 3.0, 1.3, 'Источник данных\n(Kaggle)', '#6b7280', fontsize=8.5)
box(0.3, 0.3, 3.0, 1.3, 'Аналитик /\nData Scientist', '#1b3a5c', fontsize=8.5)
box(14.7, 0.3, 3.0, 1.3, 'Пользователь\nпрогноза', '#1b3a5c', fontsize=8.5)

# контейнеры — верхний ряд (поток обработки)
box(4.5, 6.6, 2.6, 1.3, 'Приём данных', '#2b6cb0', fontsize=8.5)
box(7.5, 6.6, 2.9, 1.3, 'Хранилище\n(data/raw, incoming)', '#2b6cb0', fontsize=8.5)
box(10.8, 6.6, 3.1, 1.3, 'Подготовка данных\nи признаков', '#2b6cb0', fontsize=8.5)
box(14.3, 6.6, 2.9, 1.3, 'ML-модуль\n(обучение/инференс)', '#2b6cb0', fontsize=8.5)

# контейнеры — нижний ряд (выдача результатов)
box(5.0, 3.3, 2.7, 1.3, 'Расчёт\nпоказателей', '#2b6cb0', fontsize=8.5)
box(8.4, 3.3, 2.5, 1.3, 'Панель\n(дашборд)', '#2b6cb0', fontsize=8.5)
box(13.6, 3.3, 2.7, 1.3, 'API\n(прогноз по запросу)', '#2b6cb0', fontsize=8.5)

# связи — верхний ряд
arrow(3.3, 8.2, 4.9, 7.6, 'порции CSV', 3.9, 8.5)
arrow(7.1, 7.25, 7.5, 7.25, '', 7.3, 7.6)
arrow(10.4, 7.25, 10.8, 7.25, '', 10.6, 7.6)
arrow(13.9, 7.25, 14.3, 7.25, '', 14.1, 7.6)

# связи вниз, к нижнему ряду
arrow(15.1, 6.6, 15.1, 4.6, 'прогноз', 15.9, 5.6)
arrow(14.3, 6.8, 6.3, 4.6, 'прогнозы +\nпризнаки', 10.3, 5.9)

# связи — нижний ряд (исправлено)
arrow(7.7, 3.95, 8.4, 3.95, 'показатели', 8.05, 4.35)
arrow(8.6, 3.3, 2.3, 1.6, 'просматривает\nпанель', 5.3, 2.2)
arrow(15.1, 3.3, 16.2, 1.6, 'прогноз цены\n(API)', 16.6, 2.6)

plt.tight_layout()
plt.savefig('figures/c4_container.png', dpi=120, facecolor='white')
print("Сохранено: figures/c4_container.png")
