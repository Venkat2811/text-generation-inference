import asyncio
import os
import torch
import time
import signal

from grpc import aio
from loguru import logger

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from text_generation_server.cache import Cache
from text_generation_server.interceptor import ExceptionInterceptor
from text_generation_server.models import Model, get_model
from text_generation_server.models.pali_gemma import PaliGemmaBatch
from text_generation_server.models.vlm_causal_lm import (
    VlmCausalLMBatch,
)
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.tracing import UDSOpenTelemetryAioServerInterceptor
from text_generation_server.models.idefics_causal_lm import IdeficsCausalLMBatch
<<<<<<< Updated upstream
from text_generation_server.models.globals import set_model_id
=======
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent import futures
import grpc
>>>>>>> Stashed changes


class SignalHandler:
    KEEP_PROCESSING = True

    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print(f"Exiting gracefully: Signal {signum}")
        self.KEEP_PROCESSING = False


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(
        self,
        model: Model,
        cache: Cache,
        quantize: Optional[str],
        server_urls: List[str],
    ):
        self.cache = cache
        self.model = model
        self.quantize = quantize
        self.server_urls = server_urls
        # For some reason, inference_mode does not work well with GLOO which we use on CPU
        if model.device.type == "cuda":
            # Force inference mode for the lifetime of TextGenerationService
            self._inference_mode_raii_guard = torch._C._InferenceMode(True)
        self.executor = ThreadPoolExecutor(max_workers=10)  # Adjust max_workers based on your application's requirements

    # async def _run_in_thread(self, coro):
    #     loop = asyncio.get_running_loop()
    #     result = await loop.run_in_executor(self.executor, self._run_coroutine, coro)
    #     return result
    
    # def _run_coroutine(self, coro):
    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     try:
    #         result = loop.run_until_complete(coro)
    #     finally:
    #         loop.close()
    #     return result


    async def Info(self, request, context):
        return self.model.info

    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return generate_pb2.HealthResponse()

    async def ServiceDiscovery(self, request, context):
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    def _clear_cache(self, request, context):
        if request.HasField("id"):
            self.cache.delete(request.id)
        else:
            self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    async def ClearCache(self, request, context):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._clear_cache, request, context)
        return result
        #return self._run_in_thread(self._clear_cache(request, context))

    def _filter_batch(self, request, context):
        batch = self.cache.pop(request.batch_id)
        if batch is None:
            raise ValueError(f"Batch ID {request.batch_id} not found in cache.")
        filtered_batch = batch.filter(request.request_ids)
        self.cache.set(filtered_batch)

        return generate_pb2.FilterBatchResponse(batch=filtered_batch.to_pb())
    
    async def FilterBatch(self, request, context):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._filter_batch, request, context)
        return result
        # return self._run_in_thread(self._filter_batch(request, context))

    def _warmup(self, request, context):
        if self.quantize == "gptq":
            try:
                # When using GPTQ, Exllama kernels need some global kernels
                # For which we have the finale shapes only after the model has loaded
                # This will allocate those buffers.
                from text_generation_server.layers.gptq import (
                    create_exllama_buffers,
                    set_device,
                )

                set_device(self.model.device)
                create_exllama_buffers(request.max_prefill_tokens)
            except ImportError:
                pass

        if self.model.batch_type in {
            IdeficsCausalLMBatch,
            VlmCausalLMBatch,
            PaliGemmaBatch,
        }:  # Hack, i would rather use kwargs in the `from_pb` call
            batch = self.model.batch_type.from_pb_processor(
                request.batch,
                self.model.tokenizer,
                self.model.processor,
                self.model.model.config,
                self.model.dtype,
                self.model.device,
            )
        else:
            batch = self.model.batch_type.from_pb(
                request.batch, self.model.tokenizer, self.model.dtype, self.model.device
            )
        max_supported_total_tokens = self.model.warmup(batch)

        return generate_pb2.WarmupResponse(
            max_supported_total_tokens=max_supported_total_tokens
        )
    

    async def Warmup(self, request, context):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._warmup, request, context)
        return result
        #return self._run_in_thread(self._warmup(request, context))

    def _prefill(self, request, context):
        start = time.time_ns()
        if self.model.batch_type in {
            IdeficsCausalLMBatch,
            VlmCausalLMBatch,
            PaliGemmaBatch,
        }:  # Hack, i would rather use kwargs in the `from_pb` call
            batch = self.model.batch_type.from_pb_processor(
                request.batch,
                self.model.tokenizer,
                self.model.processor,
                self.model.model.config,
                self.model.dtype,
                self.model.device,
            )
        else:
            batch = self.model.batch_type.from_pb(
                request.batch, self.model.tokenizer, self.model.dtype, self.model.device
            )

        generations, next_batch, timings = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.PrefillResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )
    
    async def Prefill(self, request, context):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._prefill, request, context)
        return result
        # return self._run_in_thread(self._prefill(request, context))

    def _decode(self, request, context):
        start = time.time_ns()
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        batches = []
        for batch_pb in request.batches:
            batch = self.cache.pop(batch_pb.id)
            if batch is None:
                raise ValueError(f"Batch ID {batch_pb.id} not found in cache.")
            batches.append(batch)

        if len(batches) == 0:
            raise ValueError("All batches are empty")

        if len(batches) > 1:
            start_concat = time.time_ns()
            batch = self.model.batch_type.concatenate(batches)
            concat_ns = time.time_ns() - start_concat
        else:
            batch = batches[0]
            concat_ns = None

        generations, next_batch, timings = self.model.generate_token(batch)
        self.cache.set(next_batch)

        return generate_pb2.DecodeResponse(
            generations=[generation.to_pb() for generation in generations],
            batch=next_batch.to_pb() if next_batch else None,
            concat_ns=concat_ns,
            forward_ns=timings[0],
            decode_ns=timings[1],
            total_ns=time.time_ns() - start,
        )

    async def Decode(self, request, context):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self.executor, self._decode, request, context)
        return result
        # return self._run_in_thread(self._decode(request, context))


def serve(
    model_id: str,
    revision: Optional[str],
    sharded: bool,
    quantize: Optional[str],
    speculate: Optional[int],
    dtype: Optional[str],
    trust_remote_code: bool,
    uds_path: Path,
):
    async def serve_inner(
        model_id: str,
        revision: Optional[str],
        sharded: bool = False,
        quantize: Optional[str] = None,
        speculate: Optional[int] = None,
        dtype: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        unix_socket_template = "unix://{}-{}"
        if sharded:
            server_urls = [
                unix_socket_template.format(uds_path, rank)
                for rank in range(int(os.environ["WORLD_SIZE"]))
            ]
            local_url = server_urls[int(os.environ["RANK"])]
        else:
            local_url = unix_socket_template.format(uds_path, 0)
            server_urls = [local_url]

        try:
            model = get_model(
                model_id,
                revision,
                sharded,
                quantize,
                speculate,
                dtype,
                trust_remote_code,
            )
        except Exception:
            logger.exception("Error when initializing model")
            raise

        server = aio.server(
            interceptors=[
                ExceptionInterceptor(),
                UDSOpenTelemetryAioServerInterceptor(),
            ]
        )
        generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
            TextGenerationService(model, Cache(), quantize, server_urls), server
        )
        SERVICE_NAMES = (
            generate_pb2.DESCRIPTOR.services_by_name["TextGenerationService"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)

        await server.start()

        logger.info("Server started at {}".format(local_url))
        signal_handler = SignalHandler()
        while signal_handler.KEEP_PROCESSING:
            await asyncio.sleep(0.5)

    set_model_id(model_id)
    asyncio.run(
        serve_inner(
            model_id, revision, sharded, quantize, speculate, dtype, trust_remote_code
        )
    )

def _serve(model_id: str, revision: Optional[str], sharded: bool, quantize: Optional[str],
          speculate: Optional[int], dtype: Optional[str], trust_remote_code: bool, uds_path: Path):
    signal_handler = SignalHandler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         interceptors=[ExceptionInterceptor()])

    try:
        model = get_model(model_id, revision, sharded, quantize, speculate, dtype, trust_remote_code)
    except Exception as e:
        logger.exception("Error when initializing model")
        raise e

    # Generate the list of server URLs based on sharding configuration
    unix_socket_template = "unix://{}-{}"
    if sharded:
        server_urls = [unix_socket_template.format(uds_path, rank) for rank in range(int(os.environ["WORLD_SIZE"]))]
    else:
        server_urls = [unix_socket_template.format(uds_path, 0)]  # Single URL for non-sharded setup

    service_instance = TextGenerationService(model, Cache(), quantize, server_urls)
    generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(service_instance, server)

    SERVICE_NAMES = (
        generate_pb2.DESCRIPTOR.services_by_name['TextGenerationService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    local_url = server_urls[0]  # Use the first URL as the binding address
    server.add_insecure_port(local_url)
    server.start()
    logger.info(f"Server started with threads at {local_url}")

    try:
        while signal_handler.KEEP_PROCESSING:
            import time
            time.sleep(0.5)
    finally:
        server.stop(None)  # Properly shutdown the server when done or on signal

