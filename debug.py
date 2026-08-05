"""
MT5 Diagnostic Script - Run this to verify MT5 setup.
"""
import MetaTrader5 as mt5

def diagnose():
    print("=" * 70)
    print("MT5 DIAGNOSTIC REPORT")
    print("=" * 70)
    
    # Initialize
    if not mt5.initialize():
        print("[FAIL] Cannot initialize MT5")
        print(f"Error: {mt5.last_error()}")
        return
    
    # Terminal info
    terminal = mt5.terminal_info()
    print(f"\n[TERMINAL]")
    print(f"  Connected: {terminal.connected}")
    print(f"  Trade Allowed: {terminal.trade_allowed}")
    print(f"  Trade API: {terminal.tradeapi_disabled == 0}")
    print(f"  Build: {terminal.build}")
    
    # Account info
    account = mt5.account_info()
    if account:
        print(f"\n[ACCOUNT]")
        print(f"  Login: {account.login}")
        print(f"  Server: {account.server}")
        print(f"  Balance: {account.balance:.2f} {account.currency}")
        print(f"  Trade Mode: {account.trade_mode}")
    
    # Symbol info
    symbol = "XAUUSDm"  # Change to your symbol
    symbol_info = mt5.symbol_info(symbol)
    
    if not symbol_info:
        # Try without 'm'
        symbol = "XAUUSD"
        symbol_info = mt5.symbol_info(symbol)
    
    if symbol_info:
        print(f"\n[SYMBOL: {symbol}]")
        print(f"  Visible: {symbol_info.visible}")
        print(f"  Trade Mode: {symbol_info.trade_mode}")
        print(f"  Filling Mode: {symbol_info.filling_mode}")
        print(f"    - FOK allowed: {bool(symbol_info.filling_mode & 1)}")
        print(f"    - IOC allowed: {bool(symbol_info.filling_mode & 2)}")
        print(f"    - RETURN allowed: {bool(symbol_info.filling_mode & 4)}")
        print(f"  Volume min: {symbol_info.volume_min}")
        print(f"  Volume max: {symbol_info.volume_max}")
        print(f"  Volume step: {symbol_info.volume_step}")
        print(f"  Stops level: {symbol_info.trade_stops_level}")
        print(f"  Point: {symbol_info.point}")
        print(f"  Digits: {symbol_info.digits}")
        
        # Current tick
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            print(f"\n[TICK]")
            print(f"  Bid: {tick.bid}")
            print(f"  Ask: {tick.ask}")
            print(f"  Spread: {(tick.ask - tick.bid) / symbol_info.point:.0f} points")
        
        # Open positions
        positions = mt5.positions_get(symbol=symbol)
        print(f"\n[POSITIONS]")
        print(f"  Open positions: {len(positions) if positions else 0}")
        if positions:
            for p in positions[:3]:  # Show first 3
                print(f"    - Ticket {p.ticket}: {'BUY' if p.type == 0 else 'SELL'} {p.volume} @ {p.price_open}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    
    mt5.shutdown()

if __name__ == "__main__":
    diagnose()