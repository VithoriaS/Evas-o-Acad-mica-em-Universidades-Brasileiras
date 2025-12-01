"""
==============================================================================
PREDIÇÃO DE EVASÃO ACADÊMICA - REDE NEURAL COM DADOS ORIGINAIS
==============================================================================
Autor: Vitória de Lourdes Carvalho Santos
Dataset: financeiro.csv (apenas variáveis originais - 36 features)
Objetivo: Modelo Rede Neural (MLP) baseline para comparação
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

print("🧠 PREDIÇÃO DE EVASÃO - REDE NEURAL COM DADOS ORIGINAIS")
print("=" * 70)

# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================

def carregar_dados_originais(arquivo='financeiro.csv'):
    """Carrega e prepara os dados originais (sem enriquecimento)"""
    print(f"\n📊 Carregando dados originais do {arquivo}...")
    df = pd.read_csv(arquivo, low_memory=False)
    print(f"Dataset carregado: {len(df):,} registros e {len(df.columns)} colunas")
    return df

def preparar_dados(df, amostra_size=15000):
    """Prepara dados para modelagem"""
    print(f"\n🔧 Preparando dados...")
    
    # Criar variável target (Desistente + Cancelado = Evadido)
    df['ind_evadido'] = df['dsc_situacao_aluno_curso'].str.strip().isin(['Desistente', 'Cancelado']).astype(int)
    
    features_numericas = [
        'qtd_semestres_cursados', 'perc_cursado', 'media_nota_anterior',
        'media_frequencia_anterior', 'qtd_reprovacoes_curso',
        'qtd_disc_reprov_nota_curso', 'qtd_disc_reprov_frequencia_curso',
        'idade_aluno', 'val_distancia_campus'
    ]
    
    features_categoricas = [
        'sexo_aluno', 'dsc_turno', 'dsc_forma_ingresso',
        'ind_possui_bolsa', 'ind_possui_financiamento', 'ind_inadimplente'
    ]
    
    df_clean = df.dropna(subset=features_numericas + features_categoricas + ['ind_evadido'])
    print(f"Registros após limpeza: {len(df_clean):,}")
    print(f"Taxa de evasão: {df_clean['ind_evadido'].mean():.1%}")
    
    if len(df_clean) > amostra_size:
        df_sample = df_clean.groupby('ind_evadido', group_keys=False).apply(
            lambda x: x.sample(n=int(amostra_size * len(x) / len(df_clean)), random_state=42)
        )
        print(f"\nAmostra estratificada: {len(df_sample):,} registros")
    else:
        df_sample = df_clean
    
    X = df_sample[features_numericas + features_categoricas].copy()
    y = df_sample['ind_evadido']
    
    le_dict = {}
    for col in features_categoricas:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
    
    X['performance_combinada'] = X['media_nota_anterior'] * X['media_frequencia_anterior'] / 100
    X['taxa_reprovacao'] = X['qtd_reprovacoes_curso'] / (X['qtd_semestres_cursados'] + 1)
    X['faixa_etaria'] = pd.cut(X['idade_aluno'], bins=[0, 24, 34, 44, 100], labels=[0, 1, 2, 3])
    X['faixa_etaria'] = X['faixa_etaria'].cat.codes
    
    print(f"\n✅ Dataset preparado: {X.shape[0]} amostras, {X.shape[1]} features")
    
    return X, y, features_numericas + features_categoricas + ['performance_combinada', 'taxa_reprovacao', 'faixa_etaria']

# ==============================================================================
# 2. TREINAMENTO DO MODELO REDE NEURAL
# ==============================================================================

def treinar_rede_neural(X_train, y_train, X_test, y_test):
    """Treina modelo Rede Neural (MLP)"""
    print("\n🧠 Treinando modelo Rede Neural (MLP)...")
    
    # IMPORTANTE: Normalização é crucial para Redes Neurais
    print("⚙️ Normalizando dados (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Arquitetura: [100, 50, 25] neurônios (3 camadas ocultas)")
    
    # Hiperparâmetros IDÊNTICOS ao modelo enriquecido para comparação justa
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 25),  # 3 camadas: 100 -> 50 -> 25 neurônios
        activation='relu',                  # Função de ativação ReLU
        solver='adam',                      # Otimizador Adam
        alpha=0.0001,                       # Regularização L2
        batch_size=32,                      # Mini-batch
        learning_rate='adaptive',           # Taxa de aprendizado adaptativa
        learning_rate_init=0.001,           # Taxa inicial
        max_iter=300,                       # Máximo de épocas
        early_stopping=True,                # Parada antecipada
        validation_fraction=0.1,            # 10% para validação
        n_iter_no_change=20,                # Paciência: 20 épocas sem melhora
        random_state=42,
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train)
    
    print(f"✅ Modelo treinado em {model.n_iter_} iterações")
    print(f"   Loss final: {model.loss_:.4f}")
    
    return model, scaler, X_train_scaled, X_test_scaled

# ==============================================================================
# 3. AVALIAÇÃO DO MODELO
# ==============================================================================

def avaliar_modelo(model, X_test_scaled, y_test, feature_names):
    """Avalia o desempenho do modelo"""
    print("\n📈 Avaliando modelo...")
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*70)
    print("RESULTADOS - REDE NEURAL (DADOS ORIGINAIS)")
    print("="*70)
    print(f"Acurácia:  {accuracy:.1%}")
    print(f"Precisão:  {precision:.1%}")
    print(f"Recall:    {recall:.1%}")
    print(f"F1-Score:  {f1:.1%}")
    print(f"AUC-ROC:   {auc_roc:.1%}")
    print(f"Iterações: {model.n_iter_}")
    print(f"Loss:      {model.loss_:.4f}")
    print("="*70)
    
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'loss_curve': model.loss_curve_,
        'n_iter': model.n_iter_
    }

# ==============================================================================
# 4. VISUALIZAÇÕES
# ==============================================================================

def gerar_visualizacoes(resultados):
    """Gera gráficos de análise"""
    print("\n📊 Gerando visualizações...")
    
    # 1. Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = resultados['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                xticklabels=['Não Evadido', 'Evadido'],
                yticklabels=['Não Evadido', 'Evadido'])
    plt.title('Matriz de Confusão - Rede Neural (Dados Originais)', fontsize=14, fontweight='bold')
    plt.ylabel('Valor Real')
    plt.xlabel('Valor Previsto')
    plt.tight_layout()
    plt.savefig('Matriz_Confusao_Rede_Neural_Original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Matriz de Confusão salva")
    
    # 2. Curva de Aprendizado
    plt.figure(figsize=(10, 6))
    plt.plot(resultados['loss_curve'], linewidth=2, color='darkblue')
    plt.xlabel('Iterações (Épocas)', fontsize=12, fontweight='bold')
    plt.ylabel('Loss (Função de Perda)', fontsize=12, fontweight='bold')
    plt.title('Curva de Aprendizado - Rede Neural (Dados Originais)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    final_loss = resultados['loss_curve'][-1]
    plt.annotate(f'Loss Final: {final_loss:.4f}', 
                xy=(len(resultados['loss_curve'])-1, final_loss),
                xytext=(len(resultados['loss_curve'])*0.7, final_loss*1.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('Curva_Aprendizado_Rede_Neural_Original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Curva de Aprendizado salva")
    
    # 3. Curva ROC
    fpr, tpr, _ = roc_curve(resultados['y_test'], resultados['y_pred_proba'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'Rede Neural (AUC = {resultados["auc_roc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Classificador Aleatório')
    plt.xlabel('Taxa de Falsos Positivos', fontweight='bold')
    plt.ylabel('Taxa de Verdadeiros Positivos', fontweight='bold')
    plt.title('Curva ROC - Rede Neural (Dados Originais)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Curva_ROC_Rede_Neural_Original.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Curva ROC salva")

# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    df = carregar_dados_originais()
    X, y, feature_names = preparar_dados(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Divisão treino/teste:")
    print(f"   Treino: {len(X_train):,} amostras")
    print(f"   Teste:  {len(X_test):,} amostras")
    
    model, scaler, X_train_scaled, X_test_scaled = treinar_rede_neural(
        X_train, y_train, X_test, y_test
    )
    
    resultados = avaliar_modelo(model, X_test_scaled, y_test, feature_names)
    gerar_visualizacoes(resultados)
    
    resultados_df = pd.DataFrame({
        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score', 'AUC-ROC', 'Iterações', 'Loss'],
        'Valor': [
            f"{resultados['accuracy']:.1%}",
            f"{resultados['precision']:.1%}",
            f"{resultados['recall']:.1%}",
            f"{resultados['f1']:.1%}",
            f"{resultados['auc_roc']:.1%}",
            f"{resultados['n_iter']}",
            f"{resultados['loss_curve'][-1]:.4f}"
        ]
    })
    resultados_df.to_csv('Resultados_Rede_Neural_Original.csv', index=False)
    print("\n✅ Resultados salvos em 'Resultados_Rede_Neural_Original.csv'")
    
    print("\n" + "="*70)
    print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*70)

if __name__ == "__main__":
    main()
