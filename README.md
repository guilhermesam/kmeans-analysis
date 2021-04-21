# Conjunto de Parâmetros
Os parâmetros utilizados para implementação da análise do algoritmo são: _hardware_ utilizado para execução da mesma e dimensionalidade dos dados de entrada, isto é, o tamanho da instância (registros do dataset) e número de características do conjunto de dados (colunas).
## Carga de Trabalho
A carga de trabalho para a análise foi gerada através de buscas no Kaggle, utilizando palavras-chave como "churn", "customer" e "e-commerce". Em uma análise primária, foram utilizados três conjuntos de dados, descritos a seguir:
### Dataset 1: Churn Modelling (Modelagem de retenção)
Fornece detalhes sobre um consumidor, isto é, idade, gênero, valor gasto, tempo de retenção e salário anual estimado. O dataset original pode ser acessado pelo endereço https://www.kaggle.com/shubh0799/churn-modelling.

|Score de crédito|Idade|Gênero|Tempo de posse|Salário anual estimado|
|--- |--- |--- |--- |--- |
|619|42|0|2|101348.88|
|608|41|0|1|112542.58|
|502|42|0|8|113931.57|
|699|39|0|1|93826.63|
|850|43|0|2|79084.10|

__Tamanho da instância__: 10.000 linhas

### Dataset 2: Retenção de clientes - Telco
O conjunto de dados inclui informações sobre:
- Clientes que saíram no último mês - a coluna é chamada de rotatividade;
- Serviços que cada cliente assinou - telefone, várias linhas, internet, segurança online, backup online, proteção de dispositivo, suporte técnico e streaming de TV e filmes;
- Informações da conta do cliente - há quanto tempo ele é cliente, contrato, forma de pagamento, faturamento sem papel, cobranças mensais e cobranças totais;
- Informações demográficas sobre clientes - sexo, faixa etária e se eles têm parceiros e dependentes.

O conjunto de dados pode ser acessado pelo endereço https://www.kaggle.com/blastchar/telco-customer-churn.

|Gênero|Idoso?|Dependentes|Tempo de posse|
|--- |--- |--- |--- |
|M|0|0|1|
|F|0|0|34|
|F|0|0|2|
|F|0|0|45|
|M|0|0|2|

__Tamanho da instância__: 7.043 linhas

### Dataset 3: Retenção de clientes de cartão de crédito
Este dataset armazena informações referentes a clientes de uma empresa que fornece crédito, através de cartões de crédito. O dataset original possuía 23 colunas, porém, para os propósitos desta análise, serão utilizadas 5: Idade do cliente, gênero, número de dependentes que o cliente possui, limite do seu cartão de crédito e quantidade de vezes que o cliente entrou em contato com a empresa nos últimos 12 meses. O dataset original pode ser acessado através do endereço https://www.kaggle.com/sakshigoyal7/credit-card-customers.

|Idade|Gênero|Núm. de dependentes|Limite de crédito|Número de contatos (últimos 12 meses)|
|--- |--- |--- |--- |--- |
|45|M|3|12691.0|3|
|49|F|5|8256.0|2|
|51|M|3|3418.0|0|
|40|F|4|3313.0|1|
|40|M|3|4716.0|0|

__Tamanho da instância__: 10.127 linhas

### Dataset 4: Pacientes com catéter
Para diversificar as entradas do algoritmo proposto, um quarto dataset foi utilizado, de tamanho consideravelmente menor em comparação aos três anteriores, que armazena informações referentes a pacientes que utilizam catéter, tais como: altura do paciente (em polegadas), peso do paciente (em libras) e comprimento do catéter (em centímetros).
|Altura|Peso|Comprimento do catéter|
|---|---|---|
|42.8|40.0|37|
|63.5|93.5|50|
|37.5|35.5|34|
|39.5|30.0|36|
|45.5|52.0|43|
|38.5|17.0|28|
|43.0|38.5|37|
|22.5| 8.5|20|
|37.0|33.0|34|
|23.5|9.5|30|
|33.0|21.0|38|

__Tamanho da instância__: 11 linhas

## Métricas de Desempenho
Para os propósitos do presente trabalho, as métricas utilizadas são: tempo de execução e número de funções chamadas para execução do algoritmo. Os testes foram executados utilizando o módulo cProfile, próprio da linguagem Python. Os testes também foram executados em máquinas com variadas configurações de hardware, sendo listadas a seguir:

__Máquina 1__
- Processador Intel(R) Core(TM) i3-7020U CPU @ 2.30GHz;
- 4GB de memória RAM;
- 2 núcleos por soquete;
- 2 threads por núcleo;

## Resultados
Os resultados da análise serão descritos e apresentados considerando cada máquina individualmente. O algoritmo será executado um total de quatro vezes em cada máquina, sendo uma para cada dataset.

### Máquina 1
__dataset 1__:
- Tempo de execução: 18.821 segundos;
- 22.402.782 chamadas de funções, sendo 20.802.782 delas chamadas primitivas;

__dataset 2__:
- Tempo de execução: 13.642 segundos;
- 15.779.102 chamadas de funções, sendo 14.652.222 delas chamadas primitivas;

__dataset 3__:
- Tempo de execução: 18.436 segundos;
- 22.687.262 chamadas de funções, sendo 21.066.942 delas chamadas primitivas;

__dataset 4__:
- Tempo de execução: 0.037 segundos;
- 27.422 chamadas de funções, sendo 25.662 delas chamadas primitivas;

## Melhorias futuras
- Parâmetros fundamentais do algoritmo, como número de iterações e quantidade de _clusters_ foram definidos de maneira arbitrária e os mesmos para os quatro datasets, o que influencia o resultado final do processo de clusterização e não deve ser aplicado na prática. Para as próximas iterações do trabalho, iremos buscar métodos de identificar o número ótimo de clusters e de iterações para cada conjunto de dados de modo a maximizar a eficiência e preservar e garantir a confiabilidade dos resultados.
- O algoritmo não possui, até o presente momento, uma condição de parada, atingida quando os centroids dos clusters não são mais alterados em uma etapa de treinamento. Tal adição certamente reduziria o consumo computacional ao interromper iterações desnecessárias.
