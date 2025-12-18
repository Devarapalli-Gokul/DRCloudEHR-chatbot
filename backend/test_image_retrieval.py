#!/usr/bin/env python3
"""
Test script to verify image retrieval accuracy for all animals.
Tests each query and checks if the correct image is returned.
"""
import requests
import json
from typing import Dict, Tuple

# Expected mappings: query -> expected image filename
EXPECTED_MAPPINGS: Dict[str, str] = {
    "what is fox": "page_0_img_0.png",
    "what is grey squirrel": "page_0_img_1.png",
    "what is hedgehog": "page_0_img_2.png",
    "what is sparrowhawk": "page_1_img_0.png",
    "what is green woodpecker": "page_1_img_1.png",
    "what is badger": "page_1_img_2.png",
    "what is stag beetle": "page_2_img_0.png",
    "what is red admiral butterfly": "page_2_img_1.png",
    "what is grass snake": "page_2_img_2.png",
    "what is slug": "page_3_img_0.png",
    "what is snail": "page_3_img_1.png",
    "what is rabbit": "page_3_img_2.png",
}

API_URL = "http://localhost:8000/query"

def test_image_retrieval() -> Tuple[int, int]:
    """
    Test all animal queries and verify correct image retrieval.
    Returns: (correct_count, total_count)
    """
    results = []
    correct_count = 0
    total_count = len(EXPECTED_MAPPINGS)
    
    print("=" * 80)
    print("IMAGE RETRIEVAL ACCURACY TEST")
    print("=" * 80)
    print()
    
    for query, expected_image in EXPECTED_MAPPINGS.items():
        try:
            response = requests.post(
                API_URL,
                json={"question": query},
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"✗ {query:30} -> ERROR: HTTP {response.status_code}")
                results.append((query, expected_image, None, False, f"HTTP {response.status_code}"))
                continue
            
            data = response.json()
            related_images = data.get("related_images", [])
            
            if not related_images:
                print(f"✗ {query:30} -> NO IMAGES RETURNED")
                results.append((query, expected_image, None, False, "No images returned"))
                continue
            
            # Get the first image URL from the first related image
            first_image_url = related_images[0].get("image_urls", [])
            if not first_image_url:
                print(f"✗ {query:30} -> NO IMAGE URLS IN RESPONSE")
                results.append((query, expected_image, None, False, "No image URLs"))
                continue
            
            actual_image_filename = first_image_url[0].split("/")[-1]
            is_correct = expected_image in actual_image_filename
            
            if is_correct:
                correct_count += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"{status} {query:30} -> Expected: {expected_image:20} | Got: {actual_image_filename:20}")
            
            if not is_correct:
                # Show more details about what was returned
                caption = related_images[0].get("caption", "N/A")
                score = related_images[0].get("similarity_score", "N/A")
                print(f"    Caption: {caption[:60]}...")
                print(f"    Score: {score}")
            
            results.append((query, expected_image, actual_image_filename, is_correct, None))
            
        except requests.exceptions.RequestException as e:
            print(f"✗ {query:30} -> ERROR: {str(e)}")
            results.append((query, expected_image, None, False, str(e)))
        except Exception as e:
            print(f"✗ {query:30} -> ERROR: {str(e)}")
            results.append((query, expected_image, None, False, str(e)))
    
    print()
    print("=" * 80)
    print(f"RESULTS: {correct_count}/{total_count} correct ({correct_count/total_count*100:.1f}%)")
    print("=" * 80)
    print()
    
    # Show incorrect results in detail
    incorrect = [(q, exp, act, err) for q, exp, act, correct, err in results if not correct]
    if incorrect:
        print("INCORRECT RETRIEVALS:")
        print("-" * 80)
        for query, expected, actual, error in incorrect:
            print(f"Query: {query}")
            print(f"  Expected: {expected}")
            if actual:
                print(f"  Got:      {actual}")
            if error:
                print(f"  Error:    {error}")
            print()
    
    return correct_count, total_count

if __name__ == "__main__":
    try:
        correct, total = test_image_retrieval()
        exit(0 if correct == total else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        exit(1)

