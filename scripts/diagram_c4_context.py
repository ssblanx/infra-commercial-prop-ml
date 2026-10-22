import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('C4: Диаграмма контекста', fontsize=15, weight='bold', pad=15)

def box(x, y, w, h, text, color, text_color='white', fontsize=10):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                        linewidth=0, facecolor=color)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
             color=text_color, fontsize=fontsize, wrap=True)

def arrow(x1, y1, x2, y2, label, label_pos=0.5, offset=(0, 0.15)):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                          mutation_scale=16, color='#555555', linewidth=1.4)
    ax.add_patch(a)
    mx, my = x1 + (x2 - x1) * label_pos, y1 + (y2 - y1) * label_pos
    ax.text(mx + offset[0], my + offset[1], label, ha='center', va='center',
             fontsize=8.5, color='#333333',
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))

# актёры (person)
box(0.6, 7.6, 3.2, 1.6,
    'Аналитик / Data Scientist\n(ведёт пайплайн, смотрит показатели)',
    '#1b3a5c', fontsize=9)
box(10.2, 7.6, 3.2, 1.6,
    'Пользователь прогноза\n(риелтор / инвестор)',
    '#1b3a5c', fontsize=9)

# центральная система
box(4.2, 4.0, 5.6, 2.4,
    'Система анализа влияния\nинфраструктуры на стоимость\nнедвижимости',
    '#2b6cb0', fontsize=11)

# внешний источник данных
box(4.9, 0.6, 4.2, 1.6,
    'Источник данных\n(Kaggle: Sberbank Russian\nHousing Market)',
    '#6b7280', fontsize=9)

# связи
arrow(2.2, 7.6, 5.4, 6.4, 'настраивает пайплайн,\nпросматривает панель', 0.45, offset=(-1.3, 0.3))
arrow(9.8, 6.4, 11.8, 7.6, 'запрашивает прогноз\nцены (API)', 0.55, offset=(1.3, -0.3))
arrow(7.0, 4.0, 7.0, 2.2, 'предоставляет порции\nданных (CSV)', 0.5, offset=(1.9, 0))

plt.tight_layout()
plt.savefig('figures/c4_context.png', dpi=120, facecolor='white')
print("Сохранено: figures/c4_context.png")
