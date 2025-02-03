import os

# Optional: Disable telemetry
# os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Optional: Set the OLLAMA host to a remote server
# os.environ["OLLAMA_HOST"] = "http://89.117.37.210:8080/api/chat/completions"

import asyncio
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList
from langchain_ollama import ChatOllama

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


async def run_search() -> AgentHistoryList:
    agent = Agent(
        task=prompt,
        llm=ChatOllama(
           	# base_url='http://89.117.37.210:8080/api/chat/completions',
			model="llama3.2",
            # num_ctx=32000,
            disable_streaming=True
        ),
        
        use_vision=False,
		max_failures=2,
		max_actions_per_step=2,
    )

    result = await agent.run()
    return result


async def main():
    result = await run_search()
    print("\n\n", result)


if __name__ == "__main__":
    asyncio.run(main())
