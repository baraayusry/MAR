from typing import Optional, Union, Any, Type
from pydantic import BaseModel
from strands import Agent
from strands.models import Model


class QueryExpansionAgent(Agent):
    """
    An agent that uses an LLM to enrich and expand a query
    to improve retrieval performance.
    """
    def __init__(
        self,
        model: Union[Model, str, None] = None,
        *,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        name: Optional[str] = "QueryExpansionAgent",
        description: str = "Expands queries for IR",
        default_output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            tools=tools,
            system_prompt=system_prompt ,
            name=name,
            description=description,
            **kwargs,
        )
        self.default_output_schema = default_output_schema

    def _build_task_prompt(self, query: str, prompt: str) -> str:
        
        return (
            f"{prompt}\n\n"
            "Expand the following query for better recall. "
            "Use synonyms, paraphrases, and related terms. "
            "Avoid filler or explanations.\n\n"
            f"Query: {query}"
        ).strip()

    def expand(
        self,
        query: str,
        prompt: Optional[str] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        """
        Expand a query synchronously.

        Args:
            query: Original query.
            prompt: Extra instructions for shaping expansion.
            output_schema: Optional user-defined Pydantic model. Falls back to default.
            **kwargs: Passed to strands call.

        Returns:
            An instance of the user-defined schema with expanded content.
        """
        schema = output_schema or self.default_output_schema
        if schema is None:
            raise ValueError(
                "You must provide an output_schema (Pydantic BaseModel) "
                "either at init or in the expand() call."
            )
        task_prompt = self._build_task_prompt(query, prompt)
        return self.structured_output(schema, prompt=task_prompt, **kwargs)

    async def expand_async(
        self,
        query: str,
        prompt: Optional[str] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> BaseModel:
        schema = output_schema or self.default_output_schema
        if schema is None:
            raise ValueError(
                "You must provide an output_schema (Pydantic BaseModel) "
                "either at init or in the expand_async() call."
            )
        task_prompt = self._build_task_prompt(query, prompt)
        return await self.structured_output_async(schema, prompt=task_prompt, **kwargs)

    