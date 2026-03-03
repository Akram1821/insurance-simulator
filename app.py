import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import lognorm

st.set_page_config(page_title="Smart Insurance Simulator", layout="wide")

st.title("📊 Smart Insurance Risk Simulator")
st.markdown("Simulation Monte Carlo + Recherche Opérationnelle + IA")

# ===============================
# Sidebar - Paramètres entreprise
# ===============================

st.sidebar.header("🏢 Paramètres de l'entreprise")

taille = st.sidebar.slider("Taille de l'entreprise", 1, 1000, 200)
securite = st.sidebar.slider("Niveau de sécurité (1 faible - 10 élevé)", 1, 10, 5)
secteur = st.sidebar.selectbox("Secteur d'activité", 
                                ["Industrie", "Tech", "Transport", "Santé"])

prime = st.sidebar.slider("Prime annuelle proposée (€)", 10000, 500000, 100000)

n_sim = st.sidebar.slider("Nombre de simulations", 1000, 20000, 5000)

# ===============================
# Encodage secteur
# ===============================

secteur_mapping = {
    "Industrie": 0,
    "Tech": 1,
    "Transport": 2,
    "Santé": 3
}

secteur_encoded = secteur_mapping[secteur]

# ===============================
# Création dataset fictif IA
# ===============================

np.random.seed(42)
X_train = np.random.rand(1000, 3)
y_train = (
    0.3 * X_train[:,0] +
    -0.5 * X_train[:,1] +
    0.2 * X_train[:,2] +
    np.random.normal(0,0.1,1000)
) > 0.1

model = LogisticRegression()
model.fit(X_train, y_train)

# ===============================
# Prédiction probabilité sinistre
# ===============================

X_new = np.array([[taille/1000, securite/10, secteur_encoded/3]])
proba_sinistre = model.predict_proba(X_new)[0][1]

st.subheader("🤖 Probabilité prédite de sinistre")
st.write(f"Probabilité estimée : **{proba_sinistre:.2%}**")

# ===============================
# Simulation Monte Carlo
# ===============================

sinistres = np.random.binomial(1, proba_sinistre, n_sim)

# Distribution lognormale des coûts
mu = 10
sigma = 0.8
couts = lognorm(s=sigma, scale=np.exp(mu)).rvs(n_sim)

pertes = sinistres * couts
profits = prime - pertes

esperance_profit = np.mean(profits)
variance_profit = np.var(profits)
prob_perte = np.mean(profits < 0)
VaR_95 = np.percentile(profits, 5)

# ===============================
# Affichage résultats
# ===============================

col1, col2, col3 = st.columns(3)

col1.metric("Espérance du profit", f"{esperance_profit:,.0f} €")
col2.metric("Probabilité de perte", f"{prob_perte:.2%}")
col3.metric("VaR (95%)", f"{VaR_95:,.0f} €")

# ===============================
# Histogramme
# ===============================

st.subheader("📈 Distribution des profits")

fig, ax = plt.subplots()
ax.hist(profits, bins=50)
ax.set_xlabel("Profit")
ax.set_ylabel("Fréquence")
st.pyplot(fig)

# ===============================
# Optimisation RO
# ===============================

st.subheader("📐 Optimisation de la prime (RO)")

primes_test = np.linspace(10000, 500000, 50)
profits_opt = []
risk_opt = []

for p in primes_test:
    profits_temp = p - pertes
    profits_opt.append(np.mean(profits_temp))
    risk_opt.append(np.mean(profits_temp < 0))

profits_opt = np.array(profits_opt)
risk_opt = np.array(risk_opt)

# Contrainte : risque < 5%
feasible = primes_test[risk_opt < 0.05]

if len(feasible) > 0:
    best_prime = feasible[np.argmax(profits_opt[risk_opt < 0.05])]
    st.success(f"Prime optimale sous contrainte risque < 5% : {best_prime:,.0f} €")
else:
    st.error("Aucune prime ne satisfait la contrainte de risque.")

# Courbe optimisation
fig2, ax2 = plt.subplots()
ax2.plot(primes_test, profits_opt)
ax2.set_xlabel("Prime")
ax2.set_ylabel("Espérance Profit")
st.pyplot(fig2)

# ===============================
# Conclusion
# ===============================

st.subheader("🧾 Décision Finale")

if prob_perte < 0.05 and esperance_profit > 0:
    st.success("✅ Décision : Assurer cette entreprise est rentable.")
else:
    st.warning("⚠️ Risque trop élevé ou profit insuffisant.")
