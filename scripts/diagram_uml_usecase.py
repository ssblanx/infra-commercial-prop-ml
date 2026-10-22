import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 16); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_aspect('equal')
ax.set_title('UML: Диаграмма вариантов использования', fontsize=15, weight='bold', pad=14)

def actor(x, y, label):
    head = plt.Circle((x, y + 1.1), 0.22, fill=False, color='black', linewidth=1.4, zorder=3)
    ax.add_patch(head)
    ax.plot([x, x], [y + 0.88, y + 0.2], color='black', linewidth=1.4, zorder=3)          # тело
    ax.plot([x - 0.35, x + 0.35], [y + 0.65, y + 0.65], color='black', linewidth=1.4, zorder=3)  # руки
    ax.plot([x, x - 0.3], [y + 0.2, y - 0.3], color='black', linewidth=1.4, zorder=3)     # нога
    ax.plot([x, x + 0.3], [y + 0.2, y - 0.3], color='black', linewidth=1.4, zorder=3)     # нога
    ax.text(x, y - 0.55, label, ha='center', va='top', fontsize=9, wrap=True)

def usecase(x, y, text, w=3.1, h=1.2, fontsize=8.5):
    e = Ellipse((x, y), w, h, facecolor='#e8eef7', edgecolor='#2b6cb0', linewidth=1.3, zorder=3)
    ax.add_patch(e)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, zorder=4)
    return x, y, w, h

def assoc(p1, p2):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='#555555', linewidth=1.1, zorder=1)

def include(p1, p2, label='<<include>>'):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=12,
                          linestyle=(0, (4, 3)), color='#888888', linewidth=1.1, zorder=1)
    ax.add_patch(a)
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    ax.text(mx, my + 0.22, label, ha='center', va='center', fontsize=7.5, color='#555555',
             style='italic', bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'), zorder=5)

# граница системы
boundary = Rectangle((3.3, 1.0), 9.4, 8.2, fill=False, linestyle='--',
                       edgecolor='#999999', linewidth=1.3, zorder=1)
ax.add_patch(boundary)
ax.text(3.5, 8.95, 'Система анализа влияния инфраструктуры на стоимость',
         fontsize=9, color='#777777', style='italic')

# актёры
actor(1.1, 4.3, 'Аналитик /\nData Scientist')
actor(14.8, 4.3, 'Пользователь\nпрогноза')

# варианты использования
uc1 = usecase(5.7, 7.7, 'Загрузить новую\nпорцию данных')
uc2 = usecase(9.9, 7.7, 'Проверить качество\nданных')
uc3 = usecase(5.7, 5.5, 'Пересчитать\nпоказатели')
uc4 = usecase(9.9, 5.5, 'Просмотреть показатели\nна панели')
uc5 = usecase(5.7, 3.1, 'Переобучить модель')
uc6 = usecase(9.9, 3.1, 'Запросить прогноз\nцены объекта')

# связи актёр — вариант использования
assoc((1.1, 5.4), (4.3, 7.4))
assoc((1.1, 5.4), (4.3, 5.5))
assoc((1.1, 5.4), (4.3, 3.3))
assoc((14.8, 5.4), (11.4, 3.2))

# include-связи
include((7.25, 7.6), (8.35, 7.65))
include((6.6, 7.1), (6.3, 6.1))

plt.tight_layout()
plt.savefig('figures/uml_usecase.png', dpi=120, facecolor='white')
print("Сохранено: figures/uml_usecase.png")
