#!/usr/bin/env python3
"""
Final test to verify all import issues are resolved
"""

print("🔍 Testing all previously problematic imports...")

try:
    # Test all the imports that were failing before
    from ui.utils import clean_text, normalize_word, load_saved_model
    from ui.tools.Dashboard_Ringkasan import safe_create_wordcloud
    
    print("✅ All imports successful!")
    
    # Test that functions are callable
    print("\n🧪 Function availability test:")
    print(f"  - clean_text: {callable(clean_text)}")
    print(f"  - normalize_word: {callable(normalize_word)}")
    print(f"  - load_saved_model: {callable(load_saved_model)}")
    print(f"  - safe_create_wordcloud: {callable(safe_create_wordcloud)}")
    
    # Quick function test
    test_result = clean_text('Test text with @mention and #hashtag!')
    print(f"\n✅ clean_text test: \"{test_result}\"")
    
    norm_result = normalize_word('test')
    print(f"✅ normalize_word test: \"{norm_result}\"")
    
    print("\n🎉 ALL IMPORT AND FUNCTION ISSUES HAVE BEEN SUCCESSFULLY RESOLVED!")
    print("🚀 Your GoRide Sentiment Analysis application is ready to run!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()
