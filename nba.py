import streamlit as st
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#função para obter o ID do jogador a partir do nome
#add depois um filtro de jogadores por time
def get_player_id(player_name):
    player_list = players.find_players_by_full_name(player_name)
    if player_list:
        return player_list[0]['id']
    else:
        return None

#função para obter a URL da imagem do jogador
def get_player_image(player_id):
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

#função para obter os dados dos jogos do jogador
def get_player_data(player_id, season):
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    return df

#função para tentar prever pts, ast e reb usando deep learning
def predict_stats(player_data):
    X = player_data[['REB', 'AST']].values
    y_pts = player_data['PTS'].values
    y_ast = player_data['AST'].values
    y_reb = player_data['REB'].values

    X_train, X_test, y_pts_train, y_pts_test = train_test_split(X, y_pts, test_size=0.2, random_state=0)
    X_train, X_test, y_ast_train, y_ast_test = train_test_split(X, y_ast, test_size=0.2, random_state=0)
    X_train, X_test, y_reb_train, y_reb_test = train_test_split(X, y_reb, test_size=0.2, random_state=0)

    def create_model():
        model = Sequential()
        model.add(Dense(64, input_dim=2, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    model_pts = create_model()
    model_ast = create_model()
    model_reb = create_model()

    history_pts = model_pts.fit(X_train, y_pts_train, epochs=50, batch_size=10, validation_data=(X_test, y_pts_test), verbose=0)
    history_ast = model_ast.fit(X_train, y_ast_train, epochs=50, batch_size=10, validation_data=(X_test, y_ast_test), verbose=0)
    history_reb = model_reb.fit(X_train, y_reb_train, epochs=50, batch_size=10, validation_data=(X_test, y_reb_test), verbose=0)

    y_pts_pred = model_pts.predict(X_test)
    y_ast_pred = model_ast.predict(X_test)
    y_reb_pred = model_reb.predict(X_test)

    mse_pts = mean_squared_error(y_pts_test, y_pts_pred)
    mse_ast = mean_squared_error(y_ast_test, y_ast_pred)
    mse_reb = mean_squared_error(y_reb_test, y_reb_pred)

    mae_pts = mean_absolute_error(y_pts_test, y_pts_pred)
    mae_ast = mean_absolute_error(y_ast_test, y_ast_pred)
    mae_reb = mean_absolute_error(y_reb_test, y_reb_pred)

    r2_pts = r2_score(y_pts_test, y_pts_pred)
    r2_ast = r2_score(y_ast_test, y_ast_pred)
    r2_reb = r2_score(y_reb_test, y_reb_pred)

    return {
        'predicted_pts': np.mean(y_pts_pred),
        'predicted_ast': np.mean(y_ast_pred),
        'predicted_reb': np.mean(y_reb_pred),
        'mse_pts': mse_pts,
        'mse_ast': mse_ast,
        'mse_reb': mse_reb,
        'mae_pts': mae_pts,
        'mae_ast': mae_ast,
        'mae_reb': mae_reb,
        'r2_pts': r2_pts,
        'r2_ast': r2_ast,
        'r2_reb': r2_reb
    }

def main():
    st.title('NBA Player Stats Dashboard')
    st.sidebar.title('Seleção de Jogador')

    #seleção de jogador
    selected_player = st.sidebar.selectbox('Selecione um Jogador:', [''] + [player['full_name'] for player in players.get_players()])

    if selected_player:
        player_id = get_player_id(selected_player)
        if player_id:
            season = '2023-24'  #temporada a ser analisada
            player_data = get_player_data(player_id, season)
            player_image_url = get_player_image(player_id)

            #calcular médias
            mean_pts = player_data['PTS'].mean()
            mean_ast = player_data['AST'].mean()
            mean_reb = player_data['REB'].mean()

            #prever pontos, assistências e rebotes
            predictions = predict_stats(player_data)

            #exibir estatísticas do jogador
            st.subheader(f'Estatísticas de {selected_player}')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.image(player_image_url, caption=selected_player)
                #exibir as previsões feita pela deep learning
                st.subheader('Previsões')
                st.write(f'Pontos: {predictions["predicted_pts"]:.2f}')
                st.write(f'Assistências: {predictions["predicted_ast"]:.2f}')
                st.write(f'Rebotes: {predictions["predicted_reb"]:.2f}')
                
                #exibir as métricas da deep learning
                st.subheader('Métricas do Modelo')
                st.write(f'MSE de Pontos: {predictions["mse_pts"]:.4f}')
                st.write(f'MAE de Pontos: {predictions["mae_pts"]:.4f}')
                st.write(f'R² de Pontos: {predictions["r2_pts"]:.4f}')
                st.write(f'MSE de Assistências: {predictions["mse_ast"]:.4f}')
                st.write(f'MAE de Assistências: {predictions["mae_ast"]:.4f}')
                st.write(f'R² de Assistências: {predictions["r2_ast"]:.4f}')
                st.write(f'MSE de Rebotes: {predictions["mse_reb"]:.4f}')
                st.write(f'MAE de Rebotes: {predictions["mae_reb"]:.4f}')
                st.write(f'R² de Rebotes: {predictions["r2_reb"]:.4f}')
                
            with col2:
                #exibir as média do jogador na temporada
                st.subheader('Médias')
                st.write(f'Pontos: {mean_pts:.2f}')
                st.write(f'Assistências: {mean_ast:.2f}')
                st.write(f'Rebotes: {mean_reb:.2f}')
                
            st.write(player_data)

if __name__ == "__main__":
    main()