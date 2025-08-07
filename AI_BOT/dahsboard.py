from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="AI Bot Trading Dashboard")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Trading Dashboard</h1>
        <div id="trades">Loading trades...</div>
        <script>
            async function fetchTrades() {
                const response = await fetch('/api/trades');
                const trades = await response.json();
                let html = '<table><tr><th>Timestamp</th><th>Pair</th><th>Action</th><th>Profit (ETH)</th></tr>';
                trades.forEach(trade => {
                    html += `<tr>
                        <td>${trade.timestamp}</td>
                        <td>${trade.pair}</td>
                        <td>${trade.action}</td>
                        <td>${trade.profit}</td>
                    </tr>`;
                });
                html += '</table>';
                document.getElementById('trades').innerHTML = html;
            }
            fetchTrades();
            setInterval(fetchTrades, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    """
    return html_content

@app.get("/api/trades")
async def get_trades():
    # In production, you would retrieve real trade data from your database or log aggregator.
    sample_trades = [
        {"timestamp": "2025-04-10 12:00:00", "pair": "WETH/USDC", "action": "Buy", "profit": "0.05"},
        {"timestamp": "2025-04-10 12:05:00", "pair": "WMATIC/USDC", "action": "Sell", "profit": "0.03"}
    ]
    return sample_trades

if __name__ == "__main__":
    uvicorn.run("dashboard:app", host="0.0.0.0", port=8000, reload=True)
