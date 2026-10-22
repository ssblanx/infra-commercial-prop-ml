import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(19, 11))
ax.set_xlim(0, 21); ax.set_ylim(0, 10.6)
ax.axis('off')
ax.set_title('UML: Диаграмма последовательности — обработка новой порции данных',
              fontsize=14, weight='bold', pad=12)

participants = [
    (1.3, 'Аналитик'), (4.3, 'Приём данных'), (7.3, 'Хранилище'),
    (10.5, 'Подготовка данных\nи признаков'), (13.5, 'ML-модуль'),
    (16.3, 'Расчёт\nпоказателей'), (19.0, 'Панель'),
]
TOP, BOTTOM = 9.8, 0.8

for x, name in participants:
    b = FancyBboxPatch((x - 1.0, TOP), 2.0, 0.65, boxstyle="round,pad=0.05",
                        linewidth=0, facecolor='#2b6cb0', zorder=3)
    ax.add_patch(b)
    ax.text(x, TOP + 0.325, name, ha='center', va='center', color='white', fontsize=8, zorder=4)
    ax.plot([x, x], [TOP, BOTTOM], linestyle=(0, (4, 3)), color='#aaaaaa', linewidth=1, zorder=1)

def msg(x1, x2, y, label, dashed=False, fontsize=7.8):
    style = (0, (4, 3)) if dashed else '-'
    a = FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=11,
                          linestyle=style, color='#333333', linewidth=1.1, zorder=2)
    ax.add_patch(a)
    mx = (x1 + x2) / 2
    ax.text(mx, y + 0.12, label, ha='center', va='bottom', fontsize=fontsize, zorder=5,
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))

def selfcall(x, y, label, fontsize=7.8):
    a = FancyArrowPatch((x, y), (x, y - 0.3), connectionstyle='arc3,rad=-1.3',
                          arrowstyle='-|>', mutation_scale=10, color='#333333', linewidth=1.1, zorder=2)
    ax.add_patch(a)
    ax.text(x + 0.55, y - 0.03, label, ha='left', va='center', fontsize=fontsize, zorder=5,
             bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none'))

A, P, H, D, M, K, N = [p[0] for p in participants]

msg(A, P, 9.3, 'загрузить порцию (batch_0N.csv)')
msg(P, H, 8.75, 'сохранить сырые данные')
msg(P, D, 8.2, 'передать порцию на обработку')
selfcall(D, 7.65, 'проверить качество (табл. 1)')

alt_top, alt_bottom = 7.1, 1.65
alt_box = Rectangle((9.3, alt_bottom), 11.0, alt_top - alt_bottom, fill=False,
                      edgecolor='#999999', linewidth=1.2, zorder=1)
ax.add_patch(alt_box)
ax.text(9.45, alt_top - 0.22, 'alt', fontsize=9, weight='bold', zorder=5,
         bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none'))
ax.text(10.15, alt_top - 0.22, '[данные прошли проверку]', fontsize=8, style='italic',
         color='#555555', zorder=5, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none'))

selfcall(D, 6.75, 'очистка, формирование признаков')
msg(D, H, 6.15, 'сохранить таблицу признаков')
msg(D, M, 5.6, 'запросить прогноз')
msg(M, D, 5.05, 'прогнозы', dashed=True)
msg(D, K, 4.5, 'прогнозы + признаки')
selfcall(K, 3.95, 'пересчитать показатели (табл. 2)')
msg(K, N, 3.35, 'обновить показатели')
msg(N, A, 2.8, 'новые значения показателей', dashed=True)

ax.plot([9.3, 20.3], [2.35, 2.35], linestyle=(0, (4, 3)), color='#999999', linewidth=1, zorder=1)
ax.text(9.45, 2.12, '[иначе: данные не прошли проверку]', fontsize=8, style='italic',
         color='#555555', zorder=5, bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none'))
msg(D, A, 1.9, 'лог ошибок, порция отклонена (F-03)', dashed=True)

plt.tight_layout()
plt.savefig('figures/uml_sequence.png', dpi=120, facecolor='white')
print("Сохранено: figures/uml_sequence.png")
