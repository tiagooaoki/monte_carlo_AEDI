# Estudo de Caso AEDI: Risco Cibernético por Simulação de Monte Carlo (ANDI)

**Disciplina:** Análise Estatística de Dados e Informações (AEDI) — PPCA / UnB  
**Professor:** João Carlos Félix Souza  
**Grupo:** Grupo 03  
**Semente Oficial (`SEED`):** `202603`  
**Data de Entrega:** 09/12/2026  

---

## 1. Objetivo do Projeto

Este repositório implementa a solução completa do estudo de caso da **ANDI (Agência Nacional de Dados Integrados)**, um órgão federal fictício com 24 unidades de negócio que busca mensurar sua exposição a perdas cibernéticas para o exercício de 2026 e avaliar a viabilidade econômica de um pacote de controles de segurança orçado em R$ 8 milhões anuais.

O projeto é estruturado em duas Simulações de Monte Carlo (SMC) encadeadas e uma bateria de análises estatísticas:
1. **Parte A — Modelo Gerador (SMC dos Dados):** Reconstrução estocástica de 36 meses de histórico (janeiro/2023 a dezembro/2025) nas etapas A1 a A5, gerando cinco bases de dados oficiais (`.csv`) a partir de parâmetros conhecidos (Questões 1 e 2).
2. **Parte B — Análise Estatística e Estimação:** Análise das bases simuladas tratando-as como dados observacionais reais (Capítulos 1 a 11), cobrindo estatística descritiva, probabilidade e Teorema de Bayes, ajustes de distribuições, inferência, testes de hipóteses, *backtesting* de Kupiec, regressão simples e múltipla, séries temporais e modelos de classificação (Questões 3 a 13).
3. **Parte C — SMC de Risco e Decisão Executiva:** Segunda Simulação de Monte Carlo (100.000 réplicas anuais para 2026) alimentada exclusivamente pelos modelos estimados na Parte B, cálculo de VaR (95% e 99%), CVaR (95%), avaliação de ROI do pacote de controles com Números Aleatórios Comuns (CRN), análise de sensibilidade, mensuração do erro de modelagem e elaboração do Parecer Executivo à Diretoria (Questões 14 e 15).

---

## 2. Estrutura do Repositório

```text
grupo_03/
├── main.py                  # Orquestrador único do pipeline (executa Partes A, B e C)
├── requirements.txt         # Dependências Python do projeto
├── README.md                # Documentação e instruções de reprodutibilidade
├── .gitignore               # Arquivos ignorados pelo Git
├── src/
│   ├── config.py            # Parâmetros centralizados e SEED = 202603
│   ├── parte_a/             # Módulos do modelo gerador (Q1 e Q2)
│   ├── parte_b/             # Módulos de análise e inferência estatística (Q3 a Q13)
│   ├── parte_c/             # Módulos da 2ª SMC de risco e decisão (Q14 e Q15)
│   └── utils/               # RNG encapsulado, criação de pastas e logging
├── data/                    # Bases geradas: unidades, ataques, perdas, painel e série
├── output/
│   ├── figures/             # Gráficos exportados para o Relatório Técnico
│   ├── tables/              # Tabelas estatísticas exportadas (.csv / .xlsx)
│   └── models/              # Artefatos e parâmetros estimados na Parte B
├── docs/                    # Relatório Técnico, Parecer Executivo e Memorial de IA
├── logs/                    # Logs de auditoria de execução com timestamp e seed
└── tests/                   # Suíte de testes automatizados (pytest)