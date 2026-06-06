"""
Subscription Management — THE REVENUE ENGINE.

Tiered pricing for the betting picks service:

  FREE TIER  ($0/mo)  — 1 pick/day, delivered via public Telegram
  BASIC TIER ($49/mo) — Full daily card, all games, Telegram + email
  PREMIUM    ($199/mo) — Live alerts, player props, parlays, CLV tracking
  ELITE      ($999/mo) — API access, custom models, portfolio tracking

Stripe integration for payment processing.
Lemon Squeezy as fallback (better affiliate tracking).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SubscriptionTier(str, Enum):
    FREE = "free"
    BASIC = "basic"       # $49/mo
    PREMIUM = "premium"   # $199/mo
    ELITE = "elite"       # $999/mo

    @property
    def price_usd(self) -> float:
        return {
            SubscriptionTier.FREE: 0.0,
            SubscriptionTier.BASIC: 49.0,
            SubscriptionTier.PREMIUM: 199.0,
            SubscriptionTier.ELITE: 999.0,
        }[self]

    @property
    def price_label(self) -> str:
        if self == SubscriptionTier.FREE:
            return "Free"
        return f"${int(self.price_usd)}/mo"

    @property
    def max_picks_per_day(self) -> int:
        return {
            SubscriptionTier.FREE: 1,
            SubscriptionTier.BASIC: 999,  # Unlimited
            SubscriptionTier.PREMIUM: 999,
            SubscriptionTier.ELITE: 999,
        }[self]

    @property
    def features(self) -> list[str]:
        return {
            SubscriptionTier.FREE: [
                "1 pick per day",
                "Public Telegram channel",
                "24-hour delay",
            ],
            SubscriptionTier.BASIC: [
                "Full daily betting card",
                "All NBA games analyzed",
                "Telegram delivery",
                "Email delivery",
                "No delay",
                "Kelly stake sizing",
            ],
            SubscriptionTier.PREMIUM: [
                "Everything in Basic",
                "Live alerts during games",
                "Player props + parlays",
                "Small league coverage",
                "CLV tracking dashboard",
                "Historical performance",
                "Discord access",
            ],
            SubscriptionTier.ELITE: [
                "Everything in Premium",
                "REST API access",
                "Custom model training",
                "Portfolio tracking",
                "Priority support",
                "Monthly strategy call",
            ],
        }[self]


@dataclass
class Subscriber:
    """A paying (or free) subscriber."""
    user_id: str
    telegram_chat_id: Optional[str] = None
    discord_id: Optional[str] = None
    email: Optional[str] = None
    tier: SubscriptionTier = SubscriptionTier.FREE
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    is_active: bool = True
    subscribed_at: str = ""
    expires_at: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(self.expires_at)
            return datetime.now() > expiry
        except (ValueError, TypeError):
            return False

    @property
    def can_receive_picks(self) -> bool:
        return self.is_active and not self.is_expired

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "telegram_chat_id": self.telegram_chat_id,
            "discord_id": self.discord_id,
            "email": self.email,
            "tier": self.tier.value,
            "stripe_customer_id": self.stripe_customer_id,
            "is_active": self.is_active,
            "subscribed_at": self.subscribed_at,
            "expires_at": self.expires_at,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  SUBSCRIPTION MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class SubscriptionManager:
    """
    Manages subscribers, tiers, and payment integration.

    Stores subscriber data in a local JSON database (scales to thousands).
    Integrates with Stripe for payment processing.

    Usage:
        sm = SubscriptionManager("data/subscribers.json")

        # Add a new subscriber
        sm.add_subscriber(
            user_id="user_123",
            telegram_chat_id="123456789",
            tier="premium",
        )

        # Check if user can access premium picks
        if sm.can_access("user_123", "premium"):
            send_picks()

        # Get daily delivery list
        for sub in sm.get_active_subscribers("basic"):
            send_telegram(sub.telegram_chat_id, daily_card)
    """

    def __init__(self, db_path: str = "data/subscribers.json"):
        self.db_path = Path(db_path)
        self.subscribers: dict[str, Subscriber] = {}
        self._load()

    def _load(self):
        """Load subscribers from disk."""
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text(encoding="utf-8"))
                for uid, sub_data in data.items():
                    sub_data["tier"] = SubscriptionTier(sub_data.get("tier", "free"))
                    self.subscribers[uid] = Subscriber(**sub_data)
                logger.info(f"Loaded {len(self.subscribers)} subscribers")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load subscribers: {e}")
                self.subscribers = {}

    def _save(self):
        """Persist subscribers to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {uid: sub.to_dict() for uid, sub in self.subscribers.items()}
        self.db_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    # ── CRUD ─────────────────────────────────────────────────────────

    def add_subscriber(
        self,
        user_id: str,
        tier: str = "free",
        telegram_chat_id: Optional[str] = None,
        discord_id: Optional[str] = None,
        email: Optional[str] = None,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        months: int = 1,
    ) -> Subscriber:
        """Add or update a subscriber."""
        tier_enum = SubscriptionTier(tier)

        now = datetime.now()
        expires = now + timedelta(days=30 * months)

        sub = Subscriber(
            user_id=user_id,
            telegram_chat_id=telegram_chat_id,
            discord_id=discord_id,
            email=email,
            tier=tier_enum,
            stripe_customer_id=stripe_customer_id,
            stripe_subscription_id=stripe_subscription_id,
            is_active=True,
            subscribed_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )

        self.subscribers[user_id] = sub
        self._save()
        logger.info(f"Subscriber {user_id} added at {tier} tier, expires {expires.date()}")
        return sub

    def update_tier(self, user_id: str, new_tier: str, months: int = 1) -> Optional[Subscriber]:
        """Upgrade/downgrade a subscriber's tier."""
        sub = self.subscribers.get(user_id)
        if not sub:
            logger.warning(f"Cannot update tier: user {user_id} not found")
            return None

        sub.tier = SubscriptionTier(new_tier)
        now = datetime.now()
        sub.expires_at = (now + timedelta(days=30 * months)).isoformat()
        sub.is_active = True
        self._save()
        logger.info(f"Subscriber {user_id} updated to {new_tier}")
        return sub

    def cancel_subscription(self, user_id: str):
        """Cancel a subscription."""
        sub = self.subscribers.get(user_id)
        if sub:
            sub.is_active = False
            self._save()
            logger.info(f"Subscriber {user_id} cancelled")

    def remove_subscriber(self, user_id: str):
        """Permanently remove a subscriber."""
        if user_id in self.subscribers:
            del self.subscribers[user_id]
            self._save()
            logger.info(f"Subscriber {user_id} removed")

    def get_subscriber(self, user_id: str) -> Optional[Subscriber]:
        return self.subscribers.get(user_id)

    def get_subscriber_by_telegram(self, chat_id: str) -> Optional[Subscriber]:
        for sub in self.subscribers.values():
            if sub.telegram_chat_id == chat_id:
                return sub
        return None

    # ── Access Control ───────────────────────────────────────────────

    def can_access(self, user_id: str, required_tier: str) -> bool:
        """Check if a user has access to a specific tier's content."""
        sub = self.subscribers.get(user_id)
        if not sub or not sub.can_receive_picks:
            return False

        required = SubscriptionTier(required_tier)
        tiers = list(SubscriptionTier)
        return tiers.index(sub.tier) >= tiers.index(required)

    def get_active_subscribers(self, min_tier: str = "free") -> list[Subscriber]:
        """Get all active subscribers at or above a tier."""
        min_tier_enum = SubscriptionTier(min_tier)
        tiers = list(SubscriptionTier)
        min_idx = tiers.index(min_tier_enum)

        result = []
        for sub in self.subscribers.values():
            if sub.can_receive_picks and tiers.index(sub.tier) >= min_idx:
                result.append(sub)
        return result

    def get_subscribers_for_telegram(self, min_tier: str = "free") -> list[Subscriber]:
        """Get subscribers who want Telegram delivery."""
        return [
            s for s in self.get_active_subscribers(min_tier)
            if s.telegram_chat_id
        ]

    # ── Stats ────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get subscriber statistics."""
        total = len(self.subscribers)
        active = sum(1 for s in self.subscribers.values() if s.can_receive_picks)
        by_tier: dict[str, int] = {}
        for t in SubscriptionTier:
            by_tier[t.value] = sum(1 for s in self.subscribers.values() if s.tier == t)

        monthly_revenue = sum(
            s.tier.price_usd for s in self.subscribers.values()
            if s.can_receive_picks and s.tier != SubscriptionTier.FREE
        )

        return {
            "total_subscribers": total,
            "active_subscribers": active,
            "by_tier": by_tier,
            "monthly_revenue_usd": monthly_revenue,
            "annual_run_rate_usd": monthly_revenue * 12,
        }

    def format_stats(self) -> str:
        stats = self.get_stats()
        lines = [
            "📊 **SUBSCRIBER STATS**",
            "━" * 30,
            f"Total: {stats['total_subscribers']}",
            f"Active: {stats['active_subscribers']}",
            "",
            "**By Tier**",
        ]
        for tier, count in stats["by_tier"].items():
            lines.append(f"  {tier}: {count}")
        lines.extend([
            "",
            f"**Revenue**",
            f"  Monthly: ${stats['monthly_revenue_usd']:,.0f}",
            f"  Annual:  ${stats['annual_run_rate_usd']:,.0f}",
        ])
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  STRIPE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════

class StripeIntegration:
    """
    Stripe payment integration for the betting picks subscription service.

    Handles:
      - Creating checkout sessions
      - Webhook event processing
      - Subscription lifecycle management

    Usage:
        stripe = StripeIntegration(api_key="sk_live_...")
        checkout_url = stripe.create_checkout("user_123", "premium")
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (
            api_key
            or os.getenv("STRIPE_SECRET_KEY")
            or os.getenv("STRIPE_API_KEY", "")
        )
        self._enabled = bool(self.api_key) and self.api_key not in ("", "your-stripe-key-here")

    def is_enabled(self) -> bool:
        return self._enabled

    def _get_price_id(self, tier: str, interval: str = "month") -> Optional[str]:
        """Get the Stripe price ID for a tier and billing interval."""
        if interval == "year":
            price_ids = {
                "basic": os.getenv("STRIPE_PRICE_BASIC_ANNUAL", ""),
                "premium": os.getenv("STRIPE_PRICE_PREMIUM_ANNUAL", ""),
                "elite": os.getenv("STRIPE_PRICE_ELITE_ANNUAL", ""),
            }
        else:
            price_ids = {
                "basic": os.getenv("STRIPE_PRICE_BASIC", ""),
                "premium": os.getenv("STRIPE_PRICE_PREMIUM", ""),
                "elite": os.getenv("STRIPE_PRICE_ELITE", ""),
            }
        return price_ids.get(tier)

    def create_checkout_session(
        self,
        user_id: str,
        tier: str,
        success_url: str = "https://yourdomain.com/success",
        cancel_url: str = "https://yourdomain.com/pricing",
        interval: str = "month",
    ) -> Optional[str]:
        """
        Create a Stripe checkout session for a new subscription.

        Args:
            user_id: Unique user identifier.
            tier: Subscription tier (basic, premium, elite).
            success_url: Redirect URL on successful payment.
            cancel_url: Redirect URL if user cancels.
            interval: Billing interval — "month" or "year".

        Returns the checkout URL (redirect the user here).
        """
        if not self._enabled:
            logger.warning("Stripe not configured. Create account at stripe.com")
            return f"https://buy.stripe.com/test?tier={tier}&user={user_id}"

        try:
            import stripe
            stripe.api_key = self.api_key

            price_id = self._get_price_id(tier, interval)
            if not price_id:
                logger.error(f"No Stripe price ID for tier {tier}, interval {interval}")
                return None

            session = stripe.checkout.Session.create(
                mode="subscription",
                line_items=[{"price": price_id, "quantity": 1}],
                metadata={"user_id": user_id, "tier": tier, "interval": interval},
                success_url=success_url,
                cancel_url=cancel_url,
            )
            return session.url

        except ImportError:
            logger.warning("stripe package not installed. Install with: pip install stripe")
            return None
        except Exception as e:
            logger.error(f"Stripe checkout failed: {e}")
            return None

    def create_customer_portal_session(
        self,
        customer_id: str,
        return_url: str = "https://yourdomain.com/subscribe/manage",
    ) -> Optional[str]:
        """
        Create a Stripe Customer Portal session for managing subscriptions.

        Returns the portal URL (redirect the user here).
        Returns None if Stripe is not configured or the API call fails.
        """
        if not self._enabled or not customer_id:
            return None

        try:
            import stripe
            stripe.api_key = self.api_key

            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            return session.url

        except ImportError:
            logger.warning("stripe package not installed. Install with: pip install stripe")
            return None
        except Exception as e:
            logger.error(f"Stripe portal session creation failed: {e}")
            return None

    def process_webhook(self, payload: dict) -> Optional[dict]:
        """
        Process a Stripe webhook event.

        Handles:
          - checkout.session.completed  → new subscription
          - invoice.payment_succeeded   → renewal
          - customer.subscription.updated → tier change
          - customer.subscription.deleted → cancellation
        """
        if not self._enabled:
            return None

        event_type = payload.get("type", "")
        data = payload.get("data", {}).get("object", {})

        if event_type == "checkout.session.completed":
            metadata = data.get("metadata", {})
            customer_details = data.get("customer_details", {})
            interval = metadata.get("interval", "month")
            months = 12 if interval == "year" else 1
            return {
                "action": "subscribe",
                "user_id": metadata.get("user_id", ""),
                "tier": metadata.get("tier", "basic"),
                "email": customer_details.get("email", ""),
                "stripe_customer_id": data.get("customer", ""),
                "stripe_subscription_id": data.get("subscription", ""),
                "status": "active",
                "interval": interval,
                "months": months,
            }

        elif event_type == "customer.subscription.updated":
            metadata = data.get("metadata", {})
            items = data.get("items", {}).get("data", [])
            tier = "basic"
            for item in items:
                price = item.get("price", {})
                # Map price ID back to tier (you'd have a mapping)
                tier = price.get("metadata", {}).get("tier", "basic")

            return {
                "action": "update",
                "user_id": metadata.get("user_id", ""),
                "tier": tier,
                "stripe_subscription_id": data.get("id", ""),
                "status": data.get("status", ""),
            }

        elif event_type == "customer.subscription.deleted":
            metadata = data.get("metadata", {})
            return {
                "action": "cancel",
                "user_id": metadata.get("user_id", ""),
                "stripe_subscription_id": data.get("id", ""),
                "status": "cancelled",
            }

        return None
