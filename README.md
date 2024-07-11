# NBA Dashboard

## Português

Este é um dashboard para analisar estatísticas de jogadores da NBA usando as bibliotecas  'nba_api', 'pandas' e 'streamlit'.

## Configuração

### Pré-requisitos

- Python 3.7+
- Pip

## Explicação do Código

'main()' = Função principal que configura a interface do usuário utilizando a biblioteca StreamLit.  

'get_player_id(player_name)': Função para pegar o ID do jogador selecionado.  

'get_player_data(player_id, season)': Função para obter os dados dos jogos do jogador.  

'predict_stats(player_data)': Função para tentar prever pts, ast e reb usando deep learning.  

## Melhorias Possíveis

Adicionar Filtro de Time: Permitir que os usuários filtrem as estatísticas por time.  

Aprimorar Métricas de Aprendizado Profundo: Integrar modelos de machine learning para previsão e análise mais aprofundada das estatísticas dos jogadores.

## English

This is an interactive dashboard to analyze NBA player statistics using the nba_api, pandas, and streamlit libraries.

## Setup

### Prerequisites

- Python 3.7+
- Pip

## Code Explanation

'main()': Main function that sets up the user interface using the Streamlit library.

'get_player_id(player_name)': Function to fetch the ID of the selected player.

'get_player_data(player_id, season)': Function to obtain the player's game data.

'predict_stats(player_data)': Function to attempt predicting points, assists, and rebounds using deep learning.

## Possible Improvements

Add Team Filter: Allow users to filter statistics by team.  

Improve Deep Learning Metrics: Integrate machine learning models for more in-depth analysis and prediction of player statistics.