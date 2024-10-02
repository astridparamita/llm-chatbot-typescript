/* eslint-disable indent */
import { Embeddings } from "@langchain/core/embeddings";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import initRephraseChain, {
  RephraseQuestionInput,
} from "./chains/rephrase-question.chain";
import { BaseChatModel } from "langchain/chat_models/base";
import { RunnablePassthrough } from "@langchain/core/runnables";
import { getHistory } from "./history";
import initTools from "./tools";
import { AgentExecutor, createOpenAIFunctionsAgent } from "langchain/agents";
import { MessagesPlaceholder } from "@langchain/core/prompts";

// tag::function[]
export default async function initAgent(
  llm: BaseChatModel,
  embeddings: Embeddings,
  graph: Neo4jGraph
) {
  // Initiate tools
  const tools = await initTools(llm, embeddings, graph);
  // Pull the prompt from the hub
  // const prompt = await pull<ChatPromptTemplate>(
  //   "hwchase17/openai-functions-agent"
  // );
  const prompt = ChatPromptTemplate.fromMessages<{
    'chat_history': string, 
    agent_scratchpad: string, 
    rephrasedQuestion: string
  }>([
    ['system', `
    You are Ebert, a movie recommendation chatbot.
    Your goal is to provide movie lovers with excellent recommendations
    backed by data from Neo4j, the world's leading graph database.
  
    Respond to any questions that don't relate to movies, actors or directors
    with a joke about parrots, before asking them to ask another question
    related to the movie industry.
    `,
    ],
    [ 'human', '{rephrasedQuestion}' ],
    new MessagesPlaceholder({variableName: 'chat_history', optional: true}),
    new MessagesPlaceholder('agent_scratchpad'),
  ]);
  
  
  // Create an agent
  const agent = await createOpenAIFunctionsAgent({
    llm,
    tools,
    prompt,
  });
  // Create an agent executor
  const executor = new AgentExecutor({
    agent,
    tools,
    verbose: true, // Verbose output logs the agents _thinking_
  });
  // Create a rephrase question chain
  const rephraseQuestionChain = await initRephraseChain(llm);
  // Return a runnable passthrough
  return (
    RunnablePassthrough.assign<{ input: string; sessionId: string }, any>({
      // Get Message History
      history: async (_input, options) => {
        const history = await getHistory(
          options?.config.configurable.sessionId
        );
  
        return history;
      },
    })
    .assign({
      // Use History to rephrase the question
      rephrasedQuestion: (input: RephraseQuestionInput, config: any) =>
        rephraseQuestionChain.invoke(input, config),
    })
    // Pass to the executor
    .pipe(executor)
    .pick("output")
  );
}
// end::function[]
