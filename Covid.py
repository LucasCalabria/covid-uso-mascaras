import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Total da populacao, N.
N = 9616621

# Numeros inicias de infectados, recuperados e obitos; I0, R0, O0.
I0, R0, O0 = 	9670, 147528, 8724

# O resto da populacao eh suscetivel a doenca.
S0 = N - I0 - R0

# beta(taxa de contato), gamma (taxa de recuperacao (1/dias)), lambda (letalidade),
# alfa(% da populacao usando mascaras), mi (% da efetividade das mascaras).
beta, gamma, lambda_ = 0.49, 1./5.2, 0.0527
mi:   float
alfa: float

# Funcao que resolve as quatro derivadas do modelo criado para esse trabalho
def deriv(y, t, N, beta, gamma, lambda_, alfa, mi):
    S, I, R, O = y
    dSdt = -(beta * S * I * (1 - alfa) * mi / N) - (beta * S * I * (1 - mi) / N)
    dIdt = ((beta * S * I * (1 - alfa) * mi / N) + (beta * S * I * (1 - mi) / N)) - ((gamma * I) + (lambda_ * I))
    dRdt = gamma * I
    dOdt = lambda_ * I
    return dSdt, dIdt, dRdt, dOdt

# Funcao para plotar os graficos
def graph():
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)

    ax.plot(t, S/N, 'blue' , alpha=0.5, lw=2, label='Suscetíveis')
    ax.plot(t, I/N, 'red'  , alpha=0.5, lw=2, label='Infectados')
    ax.plot(t, R/N, 'green', alpha=0.5, lw=2, label='Recuperados com Imunidade')
    ax.plot(t, O/N, 'black', alpha=0.5, lw=2, label='Óbitos')

    ax.set_xlabel('Tempo /dias')
    ax.set_ylabel('Pessoas %')

    ax.set_ylim(0,1)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')

    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)

    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)


# Condicoes Iniciais para um modelo com Máscaras com 20% de Eficácia Usadas por 40% da População
y0 = S0, I0, R0, O0
mi   = 0.5
alfa = 0.8

# Grafico dos proximos 200 dias
t = np.linspace(0, 200, 200)

# Integracoes do Modelo
ret = odeint(deriv, y0, t, args=(N, beta, gamma, lambda_, alfa, mi))
S, I, R, O = ret.T

graph()
plt.suptitle("Modelo 20% Eficácia e 40% de Uso")
plt.savefig('20_40.pdf')
plt.show()