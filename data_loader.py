import ccxt
import pandas as pd


def carregar_dados(symbol='BTC/USDT', timeframe='1m', limit=1000): 
#Conexão com a Binance

    binance = ccxt.binance()
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=limit)

#Preparando os Dados e convertendo pra DataFrame

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 
                                      'low', 'close', 'volume'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df

