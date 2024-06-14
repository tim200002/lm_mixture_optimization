import argparse
from collections import defaultdict
import json
import logging
import os
import random
import time
import webdataset as wds
from webdataset import ShardWriter
from typing import List
import sys



# ========================================
# =           Global variables           =
# ========================================
BUFFER_MIN = 20000
SHARD_SIZE = 8192



def write_to_shard(chunks, shard_writer):
    for idx, chunk in enumerate(chunks):
        shard_writer.write({"__key__": f"{idx:012d}", "txt": str(chunk)})


def pop_random(els):
    """O(1) way to pop an element randomly from a list
    NOT THREAD SAFE!!! (so make sure we have a lock enabled)
    (also mutates the order of the list, but that's okay)
    """
    random_idx = random.randint(0, len(els) - 1)
    els[-1], els[random_idx] = els[random_idx], els[-1]
    return els.pop()
    

def shard_reader(shard_list, mixing_weights, added_tokens_set, shard_offset):
    for shard_idx, domain, shard_path in enumerate(shard_list):
        assert shard_path.endswith(".tar"), "Shard should be a tar file according to the webdataset"
        dataset = wds.WebDataset(shard_path)
        for dataset_idx, sample in enumerate(dataset):
            element = (shard_idx + shard_offset, dataset_idx)
            if element in added_tokens_set:
                continue
            if random.random() < mixing_weights[domain]:
                yield (domain, sample["txt"], element)  

def should_include_domain(domain_name, domains_count, desired_domains_count):
    if domain_name in desired_domains_count:
        return domains_count[domain_name] < desired_domains_count[domain_name]
    return True

def is_complete(domains_count, desired_domains_count):
    for domain_name, count in domains_count.items():
        if domain_name in desired_domains_count:
            if count < desired_domains_count[domain_name]:
                return False
        else:
            return False
    return True


def mix_tokenized_data(domains: List[str], manifests: List[str], mixing_weights: List[float], output_dir: str, output_token_count: int, chunk_size:int = 2048, no_readers: int = 8, log_path: str = None, log_level: str = "INFO"):

    logger = logging.getLogger("data_mixing")
    logger.setLevel(log_level)
    # add stream handler to console
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    if log_path:
        handler = logging.FileHandler(log_path)
    logger.addHandler(handler)





# Sanity checks
    assert len(manifests) == len(mixing_weights) == len(domains), "Number of manifests and mixing weights should be the same"
    start_time = time.time()

    logger.info("Input domains and desired mixing factors")
    for domain, weight in zip(domains, mixing_weights):
        logger.info(f"{domain}: {weight}")
    output_token_count_exteded = 1.5 * output_token_count + SHARD_SIZE * chunk_size # extend because sampling strategy only approximately includes the desired token count
    logger.info(f"Trying to generate {output_token_count} tokens, which is {output_token_count_exteded} tokens after extending")

    # Load all manifests
    domain_token_counts = defaultdict(int)
    shards = [] # list of all shards, format: [(domain, shard_path), ...]

    for domain, manifest_path in zip(domains, manifests):
        with open(manifest_path, "r") as f:
            for line in f:
                line = json.loads(line)
                shard_name = line["shard"]
                num_sequences = line["num_sequences"]
                shard_token_count = chunk_size * num_sequences
                domain_token_counts[domain] += shard_token_count

                manifest_dir_path = os.path.dirname(manifest_path)
                full_shard_path = os.path.join(manifest_dir_path, f"{shard_name}.tar")
                shards.append((domain, full_shard_path))
    
    # Calculate mixing weights
    mix_wights_normalized = [w / sum(mixing_weights) for w in mixing_weights]
    domain_mix_weights = {}
    for domain, weight in zip(domains, mix_wights_normalized):
        desired_token_count = int(output_token_count_exteded * weight)
        inclusion_ratio = desired_token_count / domain_token_counts[domain]

        assert inclusion_ratio <= 1, f"Token count for domain {domain} exceeds the total token count"
        domain_mix_weights[domain] = inclusion_ratio

    # Mix shards to introduce more randomness
    shards_shuffled = random.sample(shards, len(shards))

    logger.info(f"Domains and corresponding weights:")
    for domain, weight in domain_mix_weights.items():
        logger.info(f"{domain}: {weight}. Given total token count: {domain_token_counts[domain]} this results in {int(domain_token_counts[domain] * weight)} tokens")

    # # Start processing over shards
    added_tokens = set()
    output_directory = output_dir
    os.makedirs(output_directory, exist_ok=True)
    shard_writer = ShardWriter(os.path.join(output_directory, "shard-%07d.tar"), maxcount=SHARD_SIZE)
    files_per_worker = len(shards_shuffled) // no_readers
    readers = []

    domains_counter = defaultdict(int)
    while not is_complete(domains_counter, domain_mix_weights):
        for i in range(no_readers):
            start_idx = i * files_per_worker
            end_idx = (i+1) * files_per_worker if i != no_readers - 1 else len(shards_shuffled)
            reader = shard_reader(shards_shuffled[start_idx:end_idx], domain_mix_weights, added_tokens_set=added_tokens, shard_offset=start_idx)
            readers.append(reader)
        
        buffer = []
        while len(readers):
            for reader in readers:
                try:
                    domain, item, set_element = next(reader)
                    buffer.append((domain, item, set_element))
                    
                except StopIteration:
                    readers.remove(reader)
                    continue
                # catch all other exceptions and continue
                except Exception as e:
                    logger.error(f"Error: {e}")
                    continue
            
            if len(buffer) > BUFFER_MIN:
                chunk = []
                temp_domain_addition = []
                while len(chunk) < SHARD_SIZE and buffer:
                    domain, item, set_element = pop_random(buffer)
                    if should_include_domain(domain, domains_counter, domain_mix_weights):
                        temp_domain_addition.append((domain, item, set_element))
                        chunk.append(item)
                    elif is_complete(domains_counter, domain_mix_weights):
                        logger.info("All domains are completed. Aborting the current chunk")
                        break
                if len(chunk) == SHARD_SIZE:
                    write_to_shard(chunk, shard_writer)
                    for domain, item, set_element in temp_domain_addition:
                        domains_counter[domain] += 1
                        added_tokens.add(set_element)
                else:
                    logger.info("Chunk is not complete. Adding it back to the buffer")
                    
                assert len(chunk) == SHARD_SIZE, "Chunk size should be equal to the SHARD_SIZE"
                write_to_shard(chunk, shard_writer)
        
        # Process the remaining items after all readers completed
        if is_complete(domains_counter, domain_mix_weights):
            logger.info("All domains are completed. No longer necessary to process rest of buffer")
            break
        chunk = []
        while buffer:
            domain, item = pop_random(buffer)
            if should_include_domain(domain, domains_counter, domain_mix_weights):
                chunk.append(item)
                domains_counter[domain] += 1
            if len(chunk) == SHARD_SIZE:
                write_to_shard(chunk, shard_writer)
                chunk = []
    
    shard_writer.close()

    # stats
    total_number_of_tokens = sum(domains_counter.values()) * chunk_size
    total_time = time.time() - start_time
    logger.info(f"Finished! Processed {total_number_of_tokens} tokens in {total_time}. Per domain stats:")
    for domain, count in domains_counter.items():
        domain_tokens = count * chunk_size
        domain_percentage = domain_tokens / total_number_of_tokens * 100
        logger.info(f"{domain}: {domain_tokens} tokens, i.e. {domain_percentage:.2f}% of total tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", type=str, nargs="+", required=True, help="Domains to be mixed")
    parser.add_argument("--manifests", type=str, nargs="+", required=True, help="Manifest of all datasets to be mixed")
    parser.add_argument("--mixing-weights", type=float, nargs="+", required=True, help="Weights of each dataset in the mixing")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--token-count", type=int,  required=True, help="Number of output tokens")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Chunk size, i.e., number of tokens in each line of tokenized data")
    parser.add_argument("--no-readers", type=int, default=1, help="Number of workers for parallel processing")

    args = parser.parse_args()

    mix_tokenized_data(args.domains, args.manifests, args.mixing_weights, args.output_dir, args.token_count, args.chunk_size, args.no_readers)

    