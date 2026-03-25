import aiohttp
from bs4 import BeautifulSoup
import asyncio
import logging

logger = logging.getLogger(__name__)

async def fetch_moneycontrol_price(ticker_symbol: str):
    """
    Scrape live price from Moneycontrol.
    Note: Ticker symbol mapping might be needed for Moneycontrol URL.
    """
    # This is a sample implementation. Actual URL mapping depends on search.
    search_url = f"https://www.moneycontrol.com/india/stockpricequote/search?q={ticker_symbol}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(search_url, headers=headers, timeout=10) as response:
                if response.status != 200:
                    return None
                
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                # Moneycontrol uses different IDs for price, common patterns:
                price_div = soup.find('div', id='last_price') or soup.find('span', id='Nse_Prc_tick')
                if price_div:
                    price_text = price_div.text.strip().replace(',', '')
                    return float(price_text)
                
                # If redirection happened or searched for a specific page:
                price_container = soup.find('div', {'class': 'inprice_box'})
                if price_container:
                    price = price_container.find('strong')
                    if price:
                        return float(price.text.strip().replace(',', ''))
                    
        except Exception as e:
            logger.error(f"Error scraping Moneycontrol for {ticker_symbol}: {e}")
            return None
    return None

async def get_live_quote(ticker: str):
    """
    Get live quote from Moneycontrol or fallback to yfinance.
    """
    price = await fetch_moneycontrol_price(ticker)
    if price:
        return {"price": price, "source": "Moneycontrol (Live)"}
    
    # Fallback to a fast estimation or placeholder (since yfinance is usually in API already)
    return None
