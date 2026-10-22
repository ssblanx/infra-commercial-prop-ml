import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Polygon, Rectangle

fig, ax = plt.subplots(figsize=(13, 14))
ax.set_xlim(0, 14.5); ax.set_ylim(0, 15.2)
ax.axis('off')
ax.set_aspect('equal')
ax.set_title('UML: Диаграмма деятельности — пайплайн обработки данных',
              fontsize=14, weight='bold', pad=14)

BLUE, ORANGE, RED = '#2b6cb0', '#d97706', '#b91c1c'

def action(x, y, w, h, text, color, fontsize=8.5):
    b = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.07",
                        linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(b)
    ax.text(x, y, text, ha='center', va='center', color='white', fontsize=fontsize, zorder=4)

def decision(x, y, hw, hh, text, fontsize=8):
    pts = [(x, y+hh), (x+hw, y), (x, y-hh), (x-hw, y)]
    ax.add_patch(Polygon(pts, closed=True, facecolor='#f3f4f6', edgecolor='#555555', linewidth=1.2, zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, zorder=4)

def end_node(x, y, r=0.18):
    ax.add_patch(Circle((x, y), r + 0.08, facecolor='none', edgecolor='black', linewidth=1.2, zorder=3))
    ax.add_patch(Circle((x, y), r, facecolor='black', zorder=3))

def arrow(x1, y1, x2, y2, label='', lx=None, ly=None, fontsize=7.8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                          color='#444444', linewidth=1.2, zorder=2,
                          connectionstyle='arc3,rad=0')
    ax.add_patch(a)
    if label:
        lx = lx if lx is not None else (x1+x2)/2
        ly = ly if ly is not None else (y1+y2)/2
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fontsize, color='#333333', zorder=5,
                 bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))

# старт
ax.add_patch(Circle((7.5, 14.5), 0.2, facecolor='black', zorder=3))
arrow(7.5, 14.3, 7.5, 13.75)

action(7.5, 13.3, 3.0, 0.9, 'Загрузка', BLUE)
arrow(7.5, 12.85, 7.5, 12.35)
action(7.5, 11.9, 3.2, 0.9, 'Проверка качества', BLUE)
arrow(7.5, 11.45, 7.5, 10.95)

decision(7.5, 10.1, 1.15, 0.75, 'прошла\nпроверку?')

# ветка "нет"
arrow(6.35, 10.1, 3.4, 10.1, '[нет]', 4.9, 10.35)
action(3.4, 9.1, 3.1, 1.0, 'Логирование ошибок,\nпорция отклонена (F-03)', RED)
arrow(3.4, 8.6, 3.4, 7.75)
end_node(3.4, 7.45)

# ветка "да" -> очистка
arrow(7.5, 9.35, 7.5, 8.8, '[да]', 8.1, 9.05)
action(7.5, 8.35, 2.7, 0.9, 'Очистка', BLUE)
arrow(7.5, 7.9, 7.5, 7.4)
action(7.5, 6.95, 3.3, 0.9, 'Формирование признаков', BLUE)
arrow(7.5, 6.5, 7.5, 6.0)

decision(7.5, 5.15, 1.35, 0.8, 'режим\nобучения?')

# ветка "да, обучение" -> вправо, затем вниз к боксу
arrow(8.85, 5.15, 11.8, 5.15, '[да]', 10.3, 5.4)
arrow(11.8, 5.15, 11.8, 4.65)
action(11.8, 4.2, 2.6, 0.9, 'Обучение модели', ORANGE)
arrow(11.8, 3.75, 11.8, 3.25)
action(11.8, 2.8, 2.6, 0.9, 'Сохранение модели', ORANGE)
arrow(11.8, 2.35, 11.8, 1.75)
end_node(11.8, 1.45)

# ветка "нет, новая порция" -> вниз
arrow(7.5, 4.35, 7.5, 3.85, '[нет]', 8.1, 4.1)
action(7.5, 3.4, 3.1, 0.9, 'Сохранение таблицы\nпризнаков', BLUE)
arrow(7.5, 2.95, 7.5, 2.35)
end_node(7.5, 2.05)

# легенда
ax.add_patch(Rectangle((0.3, 0.3), 0.35, 0.35, facecolor=BLUE, zorder=3))
ax.text(0.8, 0.475, 'общий шаг (обучение и новая порция)', fontsize=8, va='center')
ax.add_patch(Rectangle((0.3, 0.85), 0.35, 0.35, facecolor=ORANGE, zorder=3))
ax.text(0.8, 1.025, 'только при обучении', fontsize=8, va='center')
ax.add_patch(Rectangle((0.3, 1.4), 0.35, 0.35, facecolor=RED, zorder=3))
ax.text(0.8, 1.575, 'ошибка / отклонение порции', fontsize=8, va='center')

plt.tight_layout()
plt.savefig('figures/pipeline_activity.png', dpi=120, facecolor='white')
print("Сохранено: figures/pipeline_activity.png")
