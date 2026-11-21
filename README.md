# PredictMaint — Pipeline Preditivo de Manutenção Industrial (Indústria 4.0)

## Sobre o Projeto
Este projeto foi desenvolvido como avaliação do Módulo 1 de Desenvolvimento de IA para Análise Preditiva. O objetivo do sistema é construir um pipeline completo de Ciência de Dados em Python para prever quebras mecânicas em máquinas industriais (`falha_maquina = 1` para falha e `falha_maquina = 0` para funcionamento normal) a partir do histórico de leituras de sensores da fábrica, ajudando a evitar paradas inesperadas na linha de produção.

## Tecnologias e Bibliotecas Utilizadas
- **Python 3** e **Jupyter Notebook**
- **Pandas** e **NumPy:** leitura, limpeza, análise estatística e manipulação dos dados
- **Matplotlib** e **Seaborn:** criação dos gráficos da análise exploratória, boxplots e comparativo final
- **Scikit-Learn:** divisão estratificada de treino/teste, escalonamento (`StandardScaler`), modelos `KNeighborsClassifier` (KNN) e `DecisionTreeClassifier` (Árvore de Decisão) e cálculo de acurácia
- **Imbalanced-Learn (`imblearn`):** balanceamento da classe minoritária nos dados de treino usando `SMOTE`

## Arquitetura e Etapas do Pipeline
```text
[Dados Brutos CSV]
   └──► Fase 1 (EDA): dimensões, .describe(), gráficos de distribuição, alvo e correlação
   └──► Fase 2 (Data Prep): verificação de duplicatas, boxplots de outliers e imputação pela mediana
   └──► Fase 3 (Feature Engineering): criação da coluna potencia (RPM * Torque)
   └──► Fase 4 (Split e Balanceamento): remoção de colunas pós-falha, split 80/20 (stratify=y) e SMOTE no treino
   └──► Fase 5 (Escalonamento): StandardScaler apenas no KNN (fit_transform no treino / transform no teste)
   └──► Fase 6 (Ajuste de Parâmetros): variação de K (KNN) e max_depth (Árvore) para combater overfitting
   └──► Fase 7 (Veredito Final): avaliação na base de teste e escolha do modelo campeão
```

### Resumo das Fases no Notebook
1. **Fase 1 (EDA):** Inspeção das `10.000` linhas e `14` colunas, tipos de dados, `.describe()` e plotagem de 3 gráficos analíticos (comprovando o desbalanceamento de `96,61%` normal contra `3,39%` falha, assimetria na rotação e correlação de `-0,88` entre rotação e torque).
2. **Fase 2 (Data Prep):** Verificação de linhas duplicadas (`0` encontradas), análise de outliers com boxplots e preenchimento dos `500` valores nulos em quatro sensores utilizando a mediana.
3. **Fase 3 (Feature Engineering):** Criação da coluna numérica `potencia` (`velocidade_rotacao_rpm * torque_nm`) sem geração de valores nulos.
4. **Fase 4 (Divisão e Balanceamento):** Codificação da variável `tipo`, remoção de IDs e das 5 colunas de falhas específicas pós-evento (evitando *Data Leakage*), divisão em 80% treino e 20% teste com `stratify=y` e aplicação do `SMOTE` apenas no treino.
5. **Fase 5 (Escalonamento):** Aplicação do `StandardScaler` apenas nas variáveis contínuas do KNN (`fit_transform` no treino e `transform` no teste), mantendo os dados da Árvore de Decisão sem escalonamento por ser um algoritmo imune à escala.
6. **Fase 6 (Ajuste de Parâmetros):** Testes variando `K` no KNN (`3, 5, 7, 9, 11`) e `max_depth` na Árvore (`3, 5, 8, 10, None`), identificando o *overfitting* em `max_depth = None` (`99,26%` no treino com diferença de `6,16` pontos percentuais para o teste) e a estabilidade da poda em `max_depth = 10`.
7. **Fase 7 (Veredito Final):** Comparação da acurácia final na base de teste e indicação da Árvore de Decisão podada como modelo recomendado.

## Resultados Obtidos na Base de Teste (20%)
| Modelo | Configuração Selecionada | Acurácia no Treino | Acurácia no Teste |
| :--- | :--- | :---: | :---: |
| **Árvore de Decisão (Recomendado)** | `max_depth = 10` | 94,05% | **92,30%** |
| **KNN** | `n_neighbors = 3` | 96,50% | 90,55% |

## Estrutura do Repositório
```text
├── data/
│   └── manutencao_preditiva.csv
├── notebooks/
│   └── pipeline_manutencao_preditiva.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

## Como Executar o Projeto
1. Clone este repositório na sua máquina:
   ```bash
   git clone https://github.com/VanessaVPG/projeto-manutencao-preditiva.git
   cd projeto-manutencao-preditiva
   ```
2. Crie e ative um ambiente virtual Python:
   ```bash
   python -m venv .venv
   # Windows (Git Bash):
   source .venv/Scripts/activate
   # Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. Instale as bibliotecas listadas no `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
4. Abra o arquivo `notebooks/pipeline_manutencao_preditiva.ipynb` no VS Code ou Jupyter Notebook e clique em **Run All** (todos os caminhos de leitura dos dados são relativos à pasta `data/`).

## Melhorias Futuras
- Incluir métricas voltadas para bases desbalanceadas na indústria, como **Recall**, **Precision**, **F1-Score** e **Matriz de Confusão**, já que deixar uma máquina quebrar sem alerta (Falso Negativo) gera alto custo na produção.
- Criar uma nova variável de diferença térmica (`temperatura_processo_k - temperatura_ar_k`) para melhorar a detecção de falhas por dissipação de calor.
- Testar algoritmos de *Ensemble* como **Random Forest** e **XGBoost** combinados com validação cruzada estratificada (*StratifiedKFold*).
