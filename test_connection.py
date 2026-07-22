import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

login = int(os.getenv("BOT_MT5_LOGIN"))
password = os.getenv("BOT_MT5_PASSWORD")
server = os.getenv("BOT_MT5_SERVER")
path = os.getenv("BOT_MT5_PATH")

if not mt5.initialize(login=login, password=password, server=server, path=path):
    print(f"MT5 initialization failed: {mt5.last_error()}")
else:
    print("MT5 connection successful!")
    account_info = mt5.account_info()
    print(f"Account: {account_info.login}")
    print(f"Balance: {account_info.balance}")
    print(f"Server: {account_info.server}")
    mt5.shutdown()