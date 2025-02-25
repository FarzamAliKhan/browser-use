import asyncio
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

from langchain_groq import ChatGroq


# dotenv
load_dotenv()

api_key = os.getenv('DEEPSEEK_API_KEY', '')
if not api_key:
	raise ValueError('DEEPSEEK_API_KEY is not set')

webui_api_key = os.getenv('WEBUI_API_KEY', '')
if not api_key:
	raise ValueError('WEBUI_API_KEY is not set')

articles = {
	'https://www.man-es.com/company/press-releases/press-details/2024/12/09/world-s-first-vlcv-methanol-retrofit-represents-blueprint-for-future-projects',
	# 'https://www.man-es.com/company/press-releases/press-details/2024/12/03/full-scale-ammonia-engine-opens-new-chapter',
	# 'https://www.man-es.com/company/press-releases/press-details/2024/10/31/methanol-orders-advance-multi-fuel-future',
	# 'https://www.man-es.com/company/press-releases/press-details/2024/10/22/ammonia-powered-engine-to-be-developed-for-medium-speed-marine-applications',
	# 'https://www.man-es.com/company/press-releases/press-details/2024/09/24/canadian-coastguard-orders--man-32-44cr-propulsion-packages'
}

prompt = """
1. Act as the best Web Scraping Crawler agent
2. Go to the articles in {articles} one by one
3. Extract the Article Headline as Title, Date Published and article text (content) completely as it is written and return them
4. Generate summary of the article content and also extract keywords out of the article
5. export json for each article with link, title, date_published, content, summary, keywords
"""


# https://api.groq.com/openai/v1/chat/completions
# deepseek-r1-distill-llama-70b
# llama-3.2-90b-vision-preview
async def run_search():
	agent = Agent(
		task=prompt,
		# llm=ChatOpenAI(
		# 	base_url='https://api.groq.com/openai/v1/chat/completions',   
		# 	model='deepseek-r1-distill-llama-70b',
		# 	api_key=SecretStr(api_key),
		# ),
		# llm=ChatOpenAI(  #vps-webui-api
		# 	base_url='http://89.117.37.210:8080/api/chat/completions',
		# 	model='llama3.2:3b',
		# 	api_key=SecretStr(webui_api_key),
		# ),
		llm = ChatGroq(
			model="llama-3.2-11b-vision-preview",
			temperature=0.0,
			max_retries=2,
		),
		use_vision=False,
		max_failures=2,
		max_actions_per_step=2,
	)

	await agent.run()


if __name__ == '__main__':
	asyncio.run(run_search())
